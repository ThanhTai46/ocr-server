#!/usr/bin/env python3
"""OCR Server — Flask API for Tesseract OCR.
Deploy: railway run python3 ocr_server.py
"""
import sys, json, urllib.request, io, time, logging, re, os
from flask import Flask, request, jsonify
import pytesseract
from PIL import Image, ImageFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [OCR] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ocr")

app = Flask(__name__)

def has_real_text(text: str) -> bool:
    if not text or len(text) < 10:
        return False
    letters = sum(1 for c in text if c.isalpha())
    if letters / max(len(text), 1) < 0.5:
        return False
    # Check for repetitive garbage like "aaaa" or "121212"
    unique_chars = len(set(text.lower().replace(" ", "")))
    if unique_chars <= 3 and len(text) > 15:
        return False
    # Must have at least 1 word with 4+ letters, or 3 words with 3+ letters
    words = re.findall(r"[a-zA-Z]{3,}", text)
    long_words = [w for w in words if len(w) >= 4]
    if len(long_words) >= 1:
        return True
    if len(words) >= 3:
        return True
    return False

def ocr_image(image_url: str) -> str:
    log.info(f"Downloading: {image_url[:80]}...")
    t0 = time.time()
    req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        img_data = resp.read()
        log.info(f"Downloaded {len(img_data)/1024:.0f}KB in {time.time()-t0:.1f}s")

    t1 = time.time()
    img = Image.open(io.BytesIO(img_data))
    img = img.convert("L")
    img = img.filter(ImageFilter.SHARPEN)

    data = pytesseract.image_to_data(img, lang="eng", config="--psm 6 --oem 3", output_type=pytesseract.Output.DICT)

    prev_line = -1
    line_texts = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else 0
        line_num = data["line_num"][i]
        if conf >= 20 and len(text) > 1:
            if line_num != prev_line and line_texts:
                line_texts.append(" ".join(line_texts.pop()))
            line_texts.append(text)
            prev_line = line_num

    if not line_texts:
        return ""

    result = "\n".join(line_texts).strip()
    if not has_real_text(result):
        return ""

    log.info(f"OCR done in {time.time()-t1:.1f}s — {len(result)} chars")
    return result

@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    data = request.get_json()
    if not data or ("imageUrl" not in data and "imageBase64" not in data):
        return jsonify({"error": "imageUrl or imageBase64 required"}), 400
    try:
        # If base64 data is provided, use it directly (avoids download issues)
        if data.get("imageBase64"):
            import base64
            img_data = base64.b64decode(data["imageBase64"])
            text = ocr_image_from_bytes(img_data)
        else:
            text = ocr_image(data["imageUrl"])
        return jsonify({"text": text, "method": "tesseract"})
    except Exception as e:
        log.error(f"Failed: {e}")
        return jsonify({"error": str(e)}), 500

def ocr_image_from_bytes(img_data: bytes) -> str:
    """Run OCR on already-downloaded image bytes."""
    import io
    t1 = time.time()
    img = Image.open(io.BytesIO(img_data))
    img = img.convert("L")
    img = img.filter(ImageFilter.SHARPEN)

    ocr_data = pytesseract.image_to_data(img, lang="eng", config="--psm 6 --oem 3", output_type=pytesseract.Output.DICT)
    prev_line = -1
    line_texts = []
    for i, text in enumerate(ocr_data["text"]):
        text = text.strip()
        conf = int(ocr_data["conf"][i]) if ocr_data["conf"][i] != "-1" else 0
        line_num = ocr_data["line_num"][i]
        if conf >= 30 and len(text) > 1:
            if line_num != prev_line and line_texts:
                line_texts.append(" ".join(line_texts.pop()))
            line_texts.append(text)
            prev_line = line_num

    result = "\n".join(line_texts).strip() if line_texts else ""
    if result and not has_real_text(result):
        result = ""
    log.info(f"OCR done in {time.time()-t1:.1f}s — {len(result)} chars")
    return result

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
