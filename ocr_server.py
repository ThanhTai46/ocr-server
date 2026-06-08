#!/usr/bin/env python3
"""OCR Server — Tesseract OCR with Flask API.
Deploy: railway run python3 ocr_server.py
"""
import sys, json, urllib.request, io, time, logging, re, os, base64
from flask import Flask, request, jsonify
from PIL import Image, ImageFilter
import pytesseract

logging.basicConfig(level=logging.INFO, format="%(asctime)s [OCR] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ocr")

app = Flask(__name__)

COMMON_WORDS = {"a", "an", "in", "on", "at", "to", "is", "it", "of", "by", "be",
                 "he", "she", "we", "they", "me", "my", "no", "so", "go", "up",
                 "us", "or", "as", "do", "if", "am", "ex", "oh", "ok", "tv",
                 "was", "had", "has", "but", "can", "for", "the", "and", "are",
                 "not", "his", "her", "its", "all", "how", "why", "who", "did",
                 "get", "got", "out", "say", "see", "too", "two", "way", "now",
                 "may", "man", "say", "let", "him"}

def clean_text(text: str) -> str:
    cleaned = re.sub(r'(?<![a-zA-Z])([a-zA-Z]) (?:([a-zA-Z]) )+([a-zA-Z])(?![a-zA-Z])',
                     lambda m: m.group(0).replace(" ", ""), text)
    cleaned = re.sub(r'(?<![a-zA-Z])([a-zA-Z]) ([a-zA-Z])(?![a-zA-Z])',
                     r'\1\2', cleaned)

    lines = cleaned.split("\n")
    good = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        letters = sum(1 for c in line if c.isalpha())
        special = sum(1 for c in line if not c.isalnum() and not c.isspace())
        total = len(line)

        if special > max(letters, 1) * 0.6 and total > 2:
            continue

        word_letters = len(re.sub(r'[^a-zA-Z]', '', line))
        if word_letters >= 4:
            good.append(line)
            continue

        word = re.sub(r'[^a-zA-Z]', '', line).lower()
        if word in COMMON_WORDS:
            good.append(line)
            continue

        if letters >= 3 and word_letters == len(line.rstrip(".,!?;:\"'")):
            good.append(line)
            continue

    return "\n".join(good)

def has_real_text(text: str) -> bool:
    text = clean_text(text)
    if not text or len(text) < 10:
        return False
    letters = sum(1 for c in text if c.isalpha())
    if letters / max(len(text), 1) < 0.5:
        return False
    words = re.findall(r"[a-zA-Z]{4,}", text)
    if len(words) >= 1:
        return True
    words3 = re.findall(r"[a-zA-Z]{3,}", text)
    return len(words3) >= 3

def ocr_image(image_url: str) -> str:
    log.info(f"Downloading: {image_url[:80]}...")
    t0 = time.time()
    req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        img_data = resp.read()
        log.info(f"Downloaded {len(img_data)/1024:.0f}KB in {time.time()-t0:.1f}s")
    return ocr_image_from_bytes(img_data)

def ocr_image_from_bytes(img_data: bytes) -> str:
    t1 = time.time()
    img = Image.open(io.BytesIO(img_data))
    img = img.convert("L")
    img = img.filter(ImageFilter.SHARPEN)

    data = pytesseract.image_to_data(img, lang="eng", config="--psm 6 --oem 3", output_type=pytesseract.Output.DICT)

    prev_line = -1
    line_texts = []
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else 0
        if conf < 20:
            continue
        line_num = data["line_num"][i]
        if line_num != prev_line:
            if line_texts:
                line_texts.append(" ".join(line_texts.pop()))
            line_texts.append(text)
            prev_line = line_num

    if line_texts:
        line_texts.append(" ".join(line_texts.pop()))

    result = "\n".join(line_texts).strip()
    log.info(f"OCR done in {time.time()-t1:.1f}s — {len(result)} chars, {len(line_texts)} lines")
    return result

@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    data = request.get_json()
    if not data or ("imageUrl" not in data and "imageBase64" not in data):
        return jsonify({"error": "imageUrl or imageBase64 required"}), 400
    try:
        if data.get("imageBase64"):
            img_data = base64.b64decode(data["imageBase64"])
            text = ocr_image_from_bytes(img_data)
        else:
            text = ocr_image(data["imageUrl"])
        cleaned = clean_text(text)
        return jsonify({"text": cleaned if has_real_text(text) else "", "method": "tesseract"})
    except Exception as e:
        log.error(f"Failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
