#!/usr/bin/env python3
"""OCR Server — EasyOCR with Flask API.
Deploy: railway run python3 ocr_server.py
"""
import sys, json, urllib.request, io, time, logging, re, os
from flask import Flask, request, jsonify

# Import EasyOCR with warmup
import easyocr
import torch

torch.set_grad_enabled(False)
torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [OCR] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ocr")

app = Flask(__name__)

_reader = None
def get_reader():
    global _reader
    if _reader is None:
        log.info("Loading EasyOCR reader (first run may take a while)...")
        t0 = time.time()
        _reader = easyocr.Reader(["en"], gpu=False)
        log.info(f"Reader ready in {time.time()-t0:.1f}s")
    return _reader

# Warmup: preload model on startup
log.info("Warming up EasyOCR model...")
t0 = time.time()
_reader = easyocr.Reader(["en"], gpu=False)
log.info(f"EasyOCR warmed up in {time.time()-t0:.1f}s")

def has_real_text(text: str) -> bool:
    if not text or len(text.strip()) < 10:
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

    reader = get_reader()
    t1 = time.time()
    results = reader.readtext(img_data, detail=0, paragraph=True)
    text = "\n".join(results).strip()
    log.info(f"OCR done in {time.time()-t1:.1f}s — {len(text)} chars, {len(results)} blocks")
    return text

def ocr_image_from_bytes(img_data: bytes) -> str:
    reader = get_reader()
    t1 = time.time()
    results = reader.readtext(img_data, detail=0, paragraph=True)
    text = "\n".join(results).strip()
    log.info(f"OCR done in {time.time()-t1:.1f}s — {len(text)} chars, {len(results)} blocks")
    return text

@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    data = request.get_json()
    if not data or ("imageUrl" not in data and "imageBase64" not in data):
        return jsonify({"error": "imageUrl or imageBase64 required"}), 400
    try:
        import base64
        if data.get("imageBase64"):
            img_data = base64.b64decode(data["imageBase64"])
            text = ocr_image_from_bytes(img_data)
        else:
            text = ocr_image(data["imageUrl"])
        return jsonify({"text": text if has_real_text(text) else "", "method": "easyocr"})
    except Exception as e:
        log.error(f"Failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
