
import re
import time
import requests
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.corpus import stopwords
from datetime import datetime
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import threading
import hashlib
import json
import tempfile
import os
import fitz  

from collections import defaultdict

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

CSE_KEYS = [
    {"key": "AIzaSyBO_jbs_g06nD2n_F9DdMMm-QAJI6aNx-E", "cx": "109c4ebd7caca45ee"},
    {"key": "AIzaSyC1LUeswH338fvtxKAmms3Hm9HRBf88lko", "cx": "00e7ff844adbe438d"},
    {"key": "AIzaSyBDHixD00KBV98r9fOfKAqhwBt4sNXec50", "cx": "e2dc1d634907e43a1"},
    {"key": "AIzaSyCqZ5G5RlV0Nd6279EXbrvSUcEPE5qzxac", "cx": "a2e7ce19669df4d61"},
    {"key": "AIzaSyDxs-vZTjOeHmrQ0DiQJ7H9nbNkFlT6Twc", "cx": "0410614e20c86471d"},
    {"key": "AIzaSyCT1Sb0gOUrrRvM6m0awISUPbhwh-Neevg", "cx": "33597f7e6e8a8494e"},
]

LOG_FILE = 'plagiarism_log.txt'
CACHE_FILE = 'search_cache.json'

class KeyRotator:
    def __init__(self, keys):
        self.keys = keys
        self.index = 0
        self.lock = threading.Lock()

    def get_next_key(self):
        with self.lock:
            key = self.keys[self.index]
            self.index = (self.index + 1) % len(self.keys)
            return key

key_rotator = KeyRotator(CSE_KEYS)

def log_event(message):
    try:
        with threading.Lock():
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now()}] {message}\n")
                f.flush()  # Ensure immediate write
    except Exception as e:
        print(f"[Logging Failed] {e}")

def clean_text(text):
    return re.sub(r'\s+', ' ', text or '').strip()

def get_cache_key(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def load_cache():
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except:
        pass

def calculate_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def create_smart_chunks(text, chunk_size=14, overlap=6):
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk_sentences = sentences[i:i + chunk_size]
        chunk = ' '.join(chunk_sentences).strip()
        if len(chunk) > 50:
            chunks.append({
                'text': chunk,
                'start_idx': i,
                'sentence_count': len(chunk_sentences)
            })
    return chunks

def fetch_full_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        return clean_text(soup.get_text(separator=' '))[:30000]
    except Exception as e:
        log_event(f"[Fetch Error] {url}: {e}")
        return None

def extract_text_from_pdf_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            temp_path = tmp.name
        doc = fitz.open(temp_path)
        text = " ".join(page.get_text() for page in doc)
        doc.close()
        os.remove(temp_path)
        return clean_text(text)
    except Exception as e:
        log_event(f"[PDF Extract Error] {url}: {e}")
        return None

def search_chunk_parallel(chunk_info, cache):
    chunk_text = chunk_info['text']
    cache_key = get_cache_key(chunk_text)
    if cache_key in cache:
        return cache[cache_key]

    SEARCH_URL = 'https://www.googleapis.com/customsearch/v1'
    log_event(f"[Chunk Start] Processing chunk {chunk_info['start_idx']}")

    for attempt in range(len(CSE_KEYS)):
        try:
            key_info = key_rotator.get_next_key()
            query = chunk_text[:256]
            params = {
                'key': key_info['key'],
                'cx': key_info['cx'],
                'q': query,
                'num': 5,
            }

            log_event(f"[Query] Chunk {chunk_info['start_idx']} → {params['q']}")
            response = requests.get(SEARCH_URL, params=params, timeout=12)
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            data = response.json()
            if 'items' not in data or not data['items']:
                log_event(f"[No Results] Chunk {chunk_info['start_idx']} returned no items")
                continue

            matches = []
            for item in data['items']:
                snippet = clean_text(item.get('snippet', ''))
                source = item.get('link', '').strip()
                if not snippet or not source:
                    continue

                similarity_snippet = calculate_similarity(chunk_text, snippet)
                best_text = snippet
                best_similarity = similarity_snippet

                full_text = extract_text_from_pdf_url(source) if source.endswith('.pdf') else fetch_full_text_from_url(source)

                if full_text:
                    excerpt_len = min(len(chunk_text) * 2, 800)
                    for i in range(0, len(full_text) - excerpt_len, 150):
                        excerpt = full_text[i:i + excerpt_len]
                        score = calculate_similarity(chunk_text, excerpt)
                        if score > best_similarity:
                            best_similarity = score
                            best_text = excerpt

                if best_similarity > 0.2:
                    matches.append({
                        'matching_text': best_text,
                        'source': source,
                        'similarity': best_similarity,
                        'chunk_info': chunk_info
                    })
                    log_event(f"[Match] {source} | Similarity: {round(best_similarity*100, 2)}%")
                    log_event(f"[Snippet] {best_text[:150]}...")

            result = {'matches': matches, 'success': True}
            cache[cache_key] = result
            return result

        except Exception as e:
            log_event(f"[Retry {attempt+1}] Chunk {chunk_info['start_idx']} error: {e}")
            time.sleep(1.5)

    result = {'matches': [], 'success': False}
    cache[cache_key] = result
    return result

def check_plagiarism_online(input_text):
    try:
        log_event("🚀 Running advanced plagiarism check")
        return check_with_google_cse(input_text)
    except Exception as e:
        log_event(f"❌ Error: {e}")
        return {
            'copied': False,
            'sources': [],
            'percentage_copied': 0.0,
            'used_api': 'none',
            'error': str(e)
        }

def check_with_google_cse(input_text):
    input_text = clean_text(input_text)
    if not input_text or len(input_text) < 30:
        return {'copied': False, 'sources': [], 'percentage_copied': 0.0}

    cache = load_cache()
    chunks = create_smart_chunks(input_text)
    if not chunks:
        return {'copied': False, 'sources': [], 'percentage_copied': 0.0}

    all_matches = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(search_chunk_parallel, chunk, cache): chunk for chunk in chunks}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result['success']:
                    all_matches.extend(result['matches'])
                time.sleep(1)
            except Exception as e:
                log_event(f"[Chunk Thread Error] {e}")

    save_cache(cache)

    source_to_matches = defaultdict(list)
    seen_texts = set()
    total_input_words = len(input_text.split())
    total_matched_words = 0
    unique_matches = []

    for match in all_matches:
        source = match['source']
        matched_text = match['matching_text']
        key = f"{source}:{matched_text[:80]}"
        if key in seen_texts:
            continue
        seen_texts.add(key)

        contribution = len(matched_text.split()) * match['similarity']
        source_to_matches[source].append((matched_text, match['similarity']))
        total_matched_words += contribution

    for source, matches in source_to_matches.items():
        source_words = sum(len(m[0].split()) * m[1] for m in matches)
        percent = round((source_words / total_input_words) * 100, 2)
        for matched_text, similarity in matches:
            unique_matches.append({
                'source': source,
                'matching_text': matched_text,
                'similarity': similarity,
                'score': percent
            })

    percentage_copied = min(round((total_matched_words / total_input_words) * 100, 2), 100.0)
    copied = percentage_copied > 15.0

    log_event(f"✅ Final plagiarism: {percentage_copied}% from {len(source_to_matches)} source(s)")

    return {
        'copied': copied,
        'sources': unique_matches,
        'percentage_copied': percentage_copied,
        'used_api': 'crawler'
    }