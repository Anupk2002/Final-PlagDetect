

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from difflib import SequenceMatcher
import json
import logging
import threading
import hashlib
import fitz  # PyMuPDF
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG ---
CSE_KEYS = [
    {"key": "AIzaSyBO_jbs_g06nD2n_F9DdMMm-QAJI6aNx-E", "cx": "109c4ebd7caca45ee"},
    {"key": "AIzaSyC1LUeswH338fvtxKAmms3Hm9HRBf88lko", "cx": "00e7ff844adbe438d"},
    {"key": "AIzaSyBDHixD00KBV98r9fOfKAqhwBt4sNXec50", "cx": "e2dc1d634907e43a1"},
    {"key": "AIzaSyCqZ5G5RlV0Nd6279EXbrvSUcEPE5qzxac", "cx": "a2e7ce19669df4d61"},
    {"key": "AIzaSyDxs-vZTjOeHmrQ0DiQJ7H9nbNkFlT6Twc", "cx": "0410614e20c86471d"},
    {"key": "AIzaSyCT1Sb0gOUrrRvM6m0awISUPbhwh-Neevg", "cx": "33597f7e6e8a8494e"},
    {"key": "AIzaSyCamZu50ooZSCWRxkrluVt-AzF7EgG2dVo", "cx": "556ca979a05d1495d"},
    {"key": "AIzaSyD3vojhS4F5g6dc7C5VExDyQlZkQSiFFuE", "cx": "a651dad0007ac4fc8"},
    {"key": "AIzaSyDHCkEqsSphUvsQsCYnmTjh4UuZPnaWu8o", "cx": "96a02bad5521b4104"},
    {"key": "AIzaSyBa7E4etKHWbDB0KZdlHeh7YaN_W5lH_jc", "cx": "e7f273958448348fc"},
    {"key": "AIzaSyDoVi9PvyQhEYjgD9FVGOy57_LW-4gPzaY", "cx": "673795c32609e4065"},
    {"key": "AIzaSyDbfHW0ykwFNDGjMcRRYtoN2dnaV9mQsB8", "cx": "8699048b3e9144756"},
    {"key": "AIzaSyCFqWVuPkIut9laT8VbXrJOXy6seYu442o", "cx": "8699048b3e9144756"},
]

HEADERS = {'User-Agent': 'Mozilla/5.0'}
SEARCH_CACHE_FILE = "search_cache.json"
BLACKLISTED_DOMAINS = {"landacbio.ipn.mx", "example.com"}
FAILED_DOMAINS = set()
SEARCH_CACHE = {}
MAX_WORKERS = 10
SIMILARITY_THRESHOLD = 0.35

logger = logging.getLogger(__name__)
lock = threading.Lock()

# --- Load Cache if available ---
if os.path.exists(SEARCH_CACHE_FILE):
    try:
        with open(SEARCH_CACHE_FILE, 'r', encoding='utf-8') as f:
            SEARCH_CACHE = json.load(f)
    except:
        SEARCH_CACHE = {}

# --- Utilities ---
def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def similarity(a, b):
    try:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    except:
        return 0

# --- Sentence chunking ---
from nltk.tokenize.punkt import PunktSentenceTokenizer
import nltk
nltk.download('punkt', quiet=True)
tokenizer = PunktSentenceTokenizer()

def chunk_sentences(text, size=20):
    sentences = tokenizer.tokenize(text)
    for i in range(0, len(sentences), size):
        yield " ".join(sentences[i:i + size])


# --- Google CSE Search ---
def get_search_results(query, key, cx):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={requests.utils.quote(query)}&key={key}&cx={cx}"
        res = requests.get(url, headers=HEADERS, timeout=6)
        if res.status_code == 200:
            return res.json().get("items", [])
    except Exception as e:
        logger.warning(f"[Google Error] {e}")
    return []

# --- Fetch Webpage Text ---
def fetch_full_text(url):
    try:
        domain = urlparse(url).netloc
        if domain in BLACKLISTED_DOMAINS or domain in FAILED_DOMAINS:
            return None
        res = requests.get(url, headers=HEADERS, timeout=6)
        if res.status_code == 200 and "text/html" in res.headers.get("Content-Type", ""):
            soup = BeautifulSoup(res.text, "html.parser")
            return soup.get_text(separator=" ", strip=True)
    except:
        FAILED_DOMAINS.add(domain)
    return None

# --- PDF Download & Extract ---
def fetch_pdf_text(url):
    try:
        r = requests.get(url, timeout=6, stream=True)
        if r.status_code == 200 and 'application/pdf' in r.headers.get('Content-Type', ''):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
                f.flush()
                doc = fitz.open(f.name)
                text = " ".join(page.get_text() for page in doc)
                doc.close()
                os.unlink(f.name)
                return text
    except:
        pass
    return None

# --- Process a Chunk ---
def process_chunk(chunk_text):
    chunk_id = hash_text(chunk_text)
    if chunk_id in SEARCH_CACHE:
        return SEARCH_CACHE[chunk_id]

    best_result = []
    for key_info in CSE_KEYS:
        results = get_search_results(chunk_text, key_info["key"], key_info["cx"])
        if not results:
            continue

        for item in results[:3]:
            url = item.get("link")
            snippet = item.get("snippet", "")
            if not url or urlparse(url).netloc in BLACKLISTED_DOMAINS:
                continue

            full_text = None
            if url.lower().endswith(".pdf"):
                full_text = fetch_pdf_text(url)
            else:
                full_text = fetch_full_text(url)

            match_score = similarity(chunk_text, full_text or snippet)
            if match_score >= SIMILARITY_THRESHOLD:
                best_result.append({
                    "source": url,
                    "score": round(match_score * 100, 2),
                    "matching_text": full_text[:500] if full_text else snippet
                })

    with lock:
        SEARCH_CACHE[chunk_id] = best_result

    return best_result

# --- Main Plagiarism Detection Function ---
def check_plagiarism_online(input_text):
    chunks = list(chunk_sentences(input_text))
    all_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
        for future in as_completed(future_map):
            try:
                chunk_results = future.result()
                if chunk_results:
                    all_results.extend(chunk_results)
            except Exception as e:
                logger.warning(f"[Chunk Error] {e}")

    sources = {}
    for r in all_results:
        url = r["source"]
        if url not in sources or r["score"] > sources[url]["score"]:
            sources[url] = {
                "score": r["score"],
                "matching_text": r["matching_text"]
            }

    top_sources = [{
        "source": url,
        "score": round(data["score"], 2),
        "matching_text": data["matching_text"]
    } for url, data in sources.items()]

    matched_chunks = sum([1 for r in all_results if r["score"] > 40])
    copied_percent = min(100, round((matched_chunks / len(chunks)) * 100, 2))

    try:
        with open(SEARCH_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(SEARCH_CACHE, f, indent=2)
    except:
        logger.error("Failed to save cache")

    return {
        "percentage_copied": copied_percent,
        "sources": top_sources,
        "combined_snippets": all_results,
        "used_api": f"{len(chunks)} chunks, {len(CSE_KEYS)} keys"
    }
