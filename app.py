import traceback
from flask import request, send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, ListFlowable, ListItem, PageBreak
from reportlab.platypus.tableofcontents import TableOfContents
import tempfile
import os
import qrcode
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import cv2
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import base64
from html import escape
import requests
import logging
import tempfile
from io import BytesIO
from datetime import datetime
from plagiarism_detector import check_plagiarism_online
from urllib.parse import urlparse
from flask import Flask, request, render_template, send_file, url_for, flash
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from skimage.metrics import structural_similarity as ssim
import fitz  # PyMuPDF
import hashlib
import time
import json
from bs4 import BeautifulSoup
import re

def extract_text_from_pdf(path):
    import fitz
    doc = fitz.open(path)
    return " ".join(page.get_text() for page in doc)

from difflib import SequenceMatcher

def highlight_matches(original_text, sources):
    """
    Highlights matched snippets in the original text with color and hyperlinks.
    Allows fuzzy substring matching (not just exact match).
    """
    import re
    from html import escape
    from difflib import SequenceMatcher

    # Use escaped HTML for safety
    clean_text = escape(original_text)
    used = set()

    color_classes = [
        "highlight-color-1", "highlight-color-2", "highlight-color-3",
        "highlight-color-4", "highlight-color-5", "highlight-color-6", "highlight-color-7"
    ]

    def find_best_substring(text, snippet):
        """Find the best fuzzy substring match in `text` for `snippet`."""
        matcher = SequenceMatcher(None, text.lower(), snippet.lower())
        match = matcher.find_longest_match(0, len(text), 0, len(snippet))
        if match.size > 20:
            return text[match.a: match.a + match.size]
        return None

    for idx, match in enumerate(sources[:7]):
        snippet = match.get("matching_text", "").strip().replace("’", "'").replace("‘", "'")
        if not snippet or snippet in used:
            continue
        used.add(snippet)

        best_substring = find_best_substring(clean_text, snippet)

        if not best_substring:
            continue

        pattern = re.escape(best_substring)
        regex = re.compile(pattern, re.IGNORECASE)

        source_url = match.get("source", "#")
        css_class = color_classes[idx % len(color_classes)]

        wrapped = (
            f'<a href="{source_url}" target="_blank" style="text-decoration: none;">'
            f'<span class="{css_class} text-black font-semibold px-1 rounded">'
            f'{escape(best_substring)} <sup>[{idx+1}]</sup></span></a>'
        )

        clean_text, count = regex.subn(wrapped, clean_text, count=1)

    return clean_text, sources




# --- Config ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
MAX_WORKERS = os.cpu_count() or 4
BLACKLISTED_DOMAINS = {'landacbio.ipn.mx'}

# --- App Setup ---
app = Flask(__name__)
app.secret_key = 'change-me'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ORB = cv2.ORB_create(nfeatures=3000)
BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def sanitize_filename(fname):
    return os.path.basename(fname)













import base64

VISION_API_KEY = "AIzaSyCvI-bt6HLsAshutAM1I0dKn9fYlnvtaE8"  # 🔐 Replace with your actual Vision API key

def get_web_urls(image_path):
    """
    Returns top 4 image URLs using Google Vision API (full + partial + similar).
    These are direct image links usable with SSIM/ORB.
    """
    try:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        encoded = base64.b64encode(img_bytes).decode('utf-8')

        vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}"
        payload = {
            "requests": [{
                "image": {"content": encoded},
                "features": [{"type": "WEB_DETECTION"}]
            }]
        }

        response = requests.post(vision_url, json=payload)
        response.raise_for_status()
        web = response.json().get("responses", [{}])[0].get("webDetection", {})

        urls = []
        for field in ['fullMatchingImages', 'partialMatchingImages', 'visuallySimilarImages']:
            for entry in web.get(field, []):
                if 'url' in entry:
                    urls.append(entry['url'])

        return urls[:5]  # top 4 direct image URLs

    except Exception as e:
        logger.warning(f"[Vision API Error] {e}")
        return []

def download_image(url):
    domain = urlparse(url).netloc
    if domain in BLACKLISTED_DOMAINS:
        logger.info(f"[Skipped - Blacklisted Domain] {url}")
        return None, url
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        r = requests.get(url, timeout=10, headers=headers, stream=True)
        r.raise_for_status()
        
        content_type = r.headers.get('Content-Type', '')
        if 'image' not in content_type.lower():
            logger.info(f"[Skipped - Not Image] {url} (Content-Type: {content_type})")
            return None, url
        
        # Read image data
        image_data = BytesIO()
        for chunk in r.iter_content(chunk_size=8192):
            image_data.write(chunk)
        image_data.seek(0)
        
        # Open and convert image
        img = Image.open(image_data).convert('RGB')
        
        # Generate safe filename
        parsed_url = urlparse(url)
        fname = sanitize_filename(parsed_url.path.split('/')[-1])
        if not fname or not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            fname = f"comp_{hashlib.md5(url.encode()).hexdigest()[:8]}.jpg"
        
        # Ensure unique filename
        base_name, ext = os.path.splitext(fname)
        counter = 1
        while os.path.exists(os.path.join(UPLOAD_FOLDER, fname)):
            fname = f"{base_name}_{counter}{ext}"
            counter += 1
        
        path = os.path.join(UPLOAD_FOLDER, fname)
        img.save(path, 'JPEG', quality=85)
        
        logger.info(f"[Downloaded] {url} -> {path}")
        return path, url
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"[Download Failed - Network] {url}: {e}")
        return None, url
    except Exception as e:
        logger.warning(f"[Download Failed - Other] {url}: {e}")
        return None, url


def compare_images(p1, p2):
    """
    Compare two images using SSIM and ORB feature matching
    """
    try:
        img1, img2 = cv2.imread(p1), cv2.imread(p2)
        if img1 is None or img2 is None:
            logger.warning(f"[Image Read Failed] {p1 if img1 is None else p2}")
            return 0.0, 0.0
        
        # Resize images to same dimensions for comparison
        height = min(img1.shape[0], img2.shape[0], 500)  # Limit size for performance
        width = min(img1.shape[1], img2.shape[1], 500)
        
        img1_resized = cv2.resize(img1, (width, height))
        img2_resized = cv2.resize(img2, (width, height))
        
        # Convert to grayscale
        g1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        ssim_score, _ = ssim(g1, g2, full=True)
        
        # Calculate ORB feature matching
        k1, d1 = ORB.detectAndCompute(g1, None)
        k2, d2 = ORB.detectAndCompute(g2, None)
        
        orb_score = 0.0
        if d1 is not None and d2 is not None and len(d1) > 0 and len(d2) > 0:
            matches = BF.match(d1, d2)
            if matches:
                # Sort matches by distance (lower is better)
                matches = sorted(matches, key=lambda x: x.distance)
                # Use top 30% of matches or at least 10 matches
                good_matches = matches[:max(10, len(matches) // 3)]
                orb_score = len(good_matches) / max(len(k1), len(k2), 1)
        
        logger.debug(f"[Compare] SSIM: {ssim_score:.3f}, ORB: {orb_score:.3f} | {os.path.basename(p1)} vs {os.path.basename(p2)}")
        return ssim_score, orb_score
        
    except Exception as e:
        logger.error(f"[Compare Error] {e}")
        return 0.0, 0.0


def calculate_plagiarism(ssim_list, orb_list, total_urls, downloaded_count):
    """
    Calculate plagiarism score based on similarity metrics
    """
    logger.debug(f"[Calculate] SSIMs: {ssim_list}, ORBs: {orb_list}, URLs: {total_urls}, Downloaded: {downloaded_count}")
    
    if not ssim_list or not orb_list:
        return {'final': 0.0}
    
    # Get maximum similarity scores
    max_ssim = max(ssim_list)
    max_orb = max(orb_list)
    
    # Calculate content similarity (weighted combination of SSIM and ORB)
    content_similarity = 0.7 * max_ssim + 0.3 * max_orb
    
    # Web presence factor (how many similar images found online)
    web_presence = min(downloaded_count / 3.0, 1.0)  # Normalize to max 1.0
    
    # Combine factors for final score
    # Higher weight on content similarity, but web presence also matters
    final_score = (0.8 * content_similarity + 0.2 * web_presence) * 100
    
    # Apply thresholds for more realistic scoring
    if max_ssim > 0.95 or max_orb > 0.8:  # Very high similarity
        final_score = max(final_score, 85)
    elif max_ssim > 0.8 or max_orb > 0.6:  # High similarity
        final_score = max(final_score, 70)
    elif max_ssim > 0.6 or max_orb > 0.4:  # Moderate similarity
        final_score = max(final_score, 50)
    
    final_score = min(final_score, 100)  # Cap at 100%
    
    logger.info(
        f"[Plagiarism Score] Max SSIM: {max_ssim:.3f}, Max ORB: {max_orb:.3f}, "
        f"Content Sim: {content_similarity:.3f}, Web Presence: {web_presence:.3f}, "
        f"Final: {final_score:.1f}%"
    )
    
    return {'final': round(final_score, 1)}


def extract_images_from_pdf(pdf_path):
    images = []
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        for img in doc.load_page(i).get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)
            ext = base['ext']
            fname = f"page_{i}_{xref}.{ext}"
            path = os.path.join(UPLOAD_FOLDER, fname)
            with open(path, 'wb') as f:
                f.write(base['image'])
            images.append(path)
    return images


from flask import Flask, request, render_template, flash

# ... (existing imports and code remain unchanged)

@app.route('/', methods=['GET', 'POST'])
def index():
    context = {
        'image_results': [],
        'total': 0,
        'plag_count': 0,
        'overall_score': 0.0
    }

    if request.method == 'POST':
        text_input = request.form.get('text_input', '').strip()
        file = request.files.get('file')

        if not text_input and not file:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return '<p class="text-red-500">Please enter text or upload a file.</p>', 400
            else:
                flash("Please enter text or upload a file.", 'danger')
                return render_template('index.html', **context)

        # Process text input
        if text_input and not file:
            try:
                plagiarism_text_result = check_plagiarism_online(text_input)
                highlighted, top_sources = highlight_matches(text_input, plagiarism_text_result.get('sources', []))
                combined = plagiarism_text_result.get('combined_snippets', [])

                context['text_results'] = {
                    'overall': plagiarism_text_result['percentage_copied'],
                    'sources': top_sources,
                    'highlighted': highlighted,
                    'api': plagiarism_text_result.get('used_api', 'N/A')
                }
                context['total'] = 1
            except Exception as e:
                context['text_results'] = {
                    'overall': 0,
                    'sources': [],
                    'combined': combined,
                    'highlighted': 'Text check failed.'
                    
                }

        # Process file input (unchanged from original)
        elif file and allowed_file(file.filename):
            filename = sanitize_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            is_pdf = filename.lower().endswith('.pdf')
            if is_pdf:
                try:
                    extracted_text = extract_text_from_pdf(path)
                    result = check_plagiarism_online(extracted_text)
                    highlighted, top_sources = highlight_matches(extracted_text, result['sources'])
                    context['text_results'] = {
                        'overall': result['percentage_copied'],
                        'sources': top_sources,
                        'highlighted': highlighted,
                        'api': result.get('used_api', 'N/A')
                    }
                except Exception as e:
                    context['text_results'] = {
                        'overall': 0,
                        'sources': [],
                        'highlighted': 'PDF text check failed.'
                    }
                image_paths = extract_images_from_pdf(path)
            else:
                image_paths = [path]
            # ... (rest of file processing remains unchanged)

            context['total'] = len(image_paths)
            max_score = 0.0
            for p in image_paths:
                urls = get_web_urls(p)
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                    results = list(pool.map(download_image, urls))
                downloaded = [(pp, u) for pp, u in results if pp]
                s_list, o_list, details = [], [], []
                for pp, u in downloaded:
                    s, o = compare_images(p, pp)
                    s_list.append(s)
                    o_list.append(o)
                    details.append({
                        'url': u,
                        'ssim': round(s, 3),
                        'orb': round(o, 3),
                        'combined': round((0.7 * s + 0.3 * o) * 100, 1)
                    })
                score = calculate_plagiarism(s_list, o_list, len(urls), len(downloaded))['final'] if s_list and o_list else 0.0
                details.sort(key=lambda x: x['combined'], reverse=True)
                context['image_results'].append({
                    'path': url_for('static', filename=f'uploads/{os.path.basename(p)}'),
                    'score': score,
                    'details': details[:10]
                })
                if score >= 30:
                    context['plag_count'] += 1
                max_score = max(max_score, score)
            context['overall_score'] = max_score
            try:
                os.remove(path)
            except:
                pass

        # ✅ Prepare structured image data
        context['image_results_data'] = {
         'total': context.get('total', 0),
         'plag_count': context.get('plag_count', 0),
         'overall_score': context.get('overall_score', 0.0),
         'image_results': context.get('image_results', [])
          }

# ✅ Return JSON with correct structure
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
           return jsonify({
             'text_results': context.get('text_results'),
            'image_results': context['image_results_data']
             })
        else:
            return render_template('index.html', **context)

    return render_template('index.html', **context)

# ... (rest of the file remains unchanged)
from flask import request, send_file
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus.tableofcontents import TableOfContents
import matplotlib.pyplot as plt
import tempfile, os, qrcode
from datetime import datetime


def convert_html_highlight_to_reportlab(html):
    soup = BeautifulSoup(html, 'html.parser')
    for span in soup.find_all("span"):
        style = span.get("style", "")
        if "background-color:" in style:
            color = style.split("background-color:")[1].split(";")[0].strip()
            font = soup.new_tag("font", backcolor=color)
            font.string = span.text
            span.replace_with(font)
    return str(soup)

@app.route('/generate-report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        report_type = data.get('type')
        filename = data.get('filename', 'N/A')

        fd, out = tempfile.mkstemp(suffix=".pdf")
        doc = SimpleDocTemplate(out, pagesize=letter, leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        flow = []

        # Cover Page
        logo_path = os.path.join('static', 'logo.png')
        if os.path.exists(logo_path):
            flow.append(RLImage(logo_path, width=2*inch, height=2*inch))
        flow.append(Spacer(1, 20))
        flow.append(Paragraph("Plagiarism Detection Report", ParagraphStyle('CoverTitle', parent=styles['Title'], fontSize=28, alignment=1, textColor=colors.HexColor('#0e3a59'))))
        flow.append(Spacer(1, 20))
        flow.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Heading3']))
        flow.append(Paragraph(f"Report Type: {report_type.capitalize()}", styles['Heading3']))
        flow.append(Paragraph(f"File: {filename}", styles['Heading3']))
        flow.append(PageBreak())

        # Table of Contents
        toc = TableOfContents()
        toc.levelStyles = [
            ParagraphStyle(fontSize=12, name='TOCHeading1', leftIndent=20, firstLineIndent=-20, spaceBefore=5, leading=14),
            ParagraphStyle(fontSize=10, name='TOCHeading2', leftIndent=40, firstLineIndent=-20, spaceBefore=0, leading=12),
        ]
        flow.append(Paragraph("Table of Contents", styles['Title']))
        flow.append(toc)
        flow.append(PageBreak())

        # Metadata
        flow.append(Paragraph("Report Details", styles['Heading1']))
        meta = [
            ["Report Date", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ["Input Type", report_type.capitalize()],
            ["File Name", filename]
        ]
        meta_table = Table(meta, colWidths=[1.8*inch, 4.5*inch])
        meta_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        flow.append(meta_table)
        flow.append(Spacer(1, 20))

        # --- TEXT ANALYSIS (for both 'text' and 'file') ---
        if report_type in ['text', 'file'] and data.get("text_results"):
            text_data = data.get("text_results")
            score = float(text_data.get("overall", 0)) if text_data.get("overall") not in [None, "None", ""] else 0
            if text_data:
                score = float(text_data.get("overall", 0))
                flow.append(Paragraph("Text Analysis Summary", styles["Heading1"]))
                flow.append(Paragraph(f"<b>Overall Plagiarism Score:</b> {score}%", styles["Heading2"]))
                labels = ["Original", "Plagiarized"]
                sizes = [100 - score, score]
                plt.figure(figsize=(2.5, 2.5))
                plt.pie(sizes, labels=labels, colors=['#4ade80', '#f87171'], autopct='%1.1f%%', startangle=140)
                pie_path = os.path.join(tempfile.gettempdir(), 'text_pie_chart.png')
                plt.savefig(pie_path)
                plt.close()
                flow.append(RLImage(pie_path, width=2.2*inch, height=2.2*inch))
                flow.append(Spacer(1, 12))
                flow.append(Paragraph("Original Text with Highlighted Matches", styles["Heading2"]))
                highlighted = convert_html_highlight_to_reportlab(text_data.get("highlighted", "N/A"))
                flow.append(Paragraph(highlighted, ParagraphStyle('TextHighlight', parent=styles["BodyText"], fontSize=10, leading=14)))
                flow.append(Spacer(1, 12))

                sources = text_data.get("sources", [])
                if sources:
                    flow.append(Paragraph("Matched Sources (Top 7)", styles["Heading2"]))
                    for i, s in enumerate(sources[:7], 1):
                        flow.append(Paragraph(f"{i}. <a href='{s['source']}'>{s['source']}</a>", styles["Normal"]))
                        flow.append(Paragraph(f"\"{s['matching_text']}\"", styles["BodyText"]))
                        flow.append(Spacer(1, 8))
                flow.append(PageBreak())

        # --- IMAGE ANALYSIS (only if 'file' input) ---
        if report_type == "file":
            img_data = data.get("image_results")
            if isinstance(img_data, dict):
                images = img_data.get("image_results", [])
                total = int(img_data.get("total", 0))
                plag_count = int(img_data.get("plag_count", 0))
                overall_score = float(img_data.get("overall_score", 0))

                flow.append(Paragraph("Image Analysis Summary", styles["Heading1"]))
                flow.append(Paragraph(f"<b>Total Images:</b> {total} | <b>Plagiarized:</b> {plag_count} | <b>Score:</b> {overall_score}%", styles["Heading2"]))

                plt.figure(figsize=(2.5, 2.5))
                plt.pie([total - plag_count, plag_count], labels=["Original", "Plagiarized"], colors=['#60a5fa', '#f87171'], autopct='%1.1f%%', startangle=140)
                img_pie_path = os.path.join(tempfile.gettempdir(), 'img_pie_chart.png')
                plt.savefig(img_pie_path)
                plt.close()
                flow.append(RLImage(img_pie_path, width=2.2*inch, height=2.2*inch))
                flow.append(Spacer(1, 10))

                for img in images:
                    flow.append(Paragraph(f"<b>Image Match Score:</b> {img['score']}%", styles["Heading3"]))
                    img_path = os.path.join("static", img["path"].replace("/static/", ""))
                    if os.path.exists(img_path):
                        flow.append(RLImage(img_path, width=4*inch, height=4*inch))
                    for i, d in enumerate(img.get("details", [])[:4], 1):
                        flow.append(Paragraph(f"{i}. <a href='{d['url']}'>{d['url']}</a>", styles["Normal"]))
                        flow.append(Paragraph(f"SSIM: {d['ssim']}, ORB: {d['orb']}, Combined: {d['combined']}%", styles["BodyText"]))
                        flow.append(Spacer(1, 6))
                    flow.append(PageBreak())

        # FINAL CONCLUSION
        flow.append(Paragraph("Final Conclusion", styles["Heading1"]))
        tscore = data.get("text_results", {}).get("overall")
        tscore = f"{float(tscore)}%" if tscore not in [None, "None", ""] else "0%"

        tscore = data.get("text_results", {}).get("overall") if report_type in ["text", "file"] else "N/A"
        iscore = data.get("image_results", {}).get("overall_score") if report_type == "file" else "N/A"
        conclusion = f"""
        <b>Text Plagiarism:</b> {tscore}%<br/>
        <b>Image Plagiarism:</b> {iscore}%<br/>
        <br/>
        This report summarizes the findings from both textual and visual content analysis. 
        Please review the matched sources and use this report to guide ethical use and originality.
        """
        flow.append(Paragraph(conclusion, ParagraphStyle('Conclusion', parent=styles["BodyText"], fontSize=10, leading=14)))
        flow.append(Spacer(1, 20))

        # Signature & QR
        flow.append(Paragraph("<b>Authorized by:</b> Plagiarism Detection System", styles["Normal"]))
        qr = qrcode.make("https://your-auth-checker.com/verify")
        qr_path = os.path.join(tempfile.gettempdir(), "qr.png")
        qr.save(qr_path)
        flow.append(RLImage(qr_path, width=1.2*inch, height=1.2*inch))

        # Footer Function
        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 8)
            canvas.drawString(0.5*inch, 0.5*inch, f"Generated by PlagDetect | {datetime.now().strftime('%Y-%m-%d')}")
            canvas.restoreState()

        doc.build(flow, onFirstPage=footer, onLaterPages=footer)
        return send_file(out, as_attachment=True, download_name='plagiarism-report.pdf', mimetype='application/pdf')

    except Exception as e:
        return f"Error generating report: {e}", 500



@app.route('/check_online', methods=['POST'])
def check_online():
    try:
        text = request.form.get('text', '').strip()
        file = request.files.get('file')

        # Case 1: Both empty
        if not text and not file:
            return jsonify({"error": "No input provided"}), 400

        # Case 2: File upload handling
        if file and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], sanitize_filename(file.filename))
            file.save(temp_path)
            if file.filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(temp_path)
            os.remove(temp_path)

        if not text:
            return jsonify({"error": "No text could be extracted from file"}), 400

        # Run plagiarism detection
        result = check_plagiarism_online(text)
        highlighted, top_sources = highlight_matches(text, result['sources'])

        # Return in structure your frontend expects
        return jsonify({
            'text_results': {
                'overall': result.get('percentage_copied', 0.0),
                'sources': top_sources,
                'highlighted': highlighted,
                'api': result.get('used_api', 'N/A')
            }
        })

    except Exception as e:
        logger.exception("[check_online] error")
        return jsonify({"error": str(e)}), 500



def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""    
    try:
        reader = PdfReader(file_path)
        text = ''.join([page.extract_text() or '' for page in reader.pages])
        return text
    except Exception as e:
        return ''


if __name__ == '__main__':
    app.run()
