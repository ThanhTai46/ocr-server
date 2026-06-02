#!/usr/bin/env python3
"""OCR Service — extracts text from images using Tesseract.
Filters out low-confidence and garbage detections.
Usage: python3 ocr.py <image_url>
Returns: {"text": "extracted text"}
"""
import sys, json, urllib.request, io, time, logging
import pytesseract
from PIL import Image, ImageFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [OCR] %(levelname)s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
log = logging.getLogger("ocr")

def has_real_text(text: str, min_char_ratio: float = 0.6) -> bool:
    """Check if text contains real words vs random characters."""
    import re
    if not text or len(text) < 10:
        return False
    # Count alphabetic chars vs total
    letters = sum(1 for c in text if c.isalpha())
    if letters / max(len(text), 1) < min_char_ratio:
        return False
    # Must have at least one real word (3+ letters)
    words = re.findall(r"[a-zA-Z]{3,}", text)
    return len(words) >= 2

def ocr_image(image_url: str) -> str:
    log.info(f"Downloading: {image_url[:80]}...")
    t0 = time.time()
    req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        img_data = resp.read()
        log.info(f"Downloaded {len(img_data)/1024:.0f}KB in {time.time()-t0:.1f}s")

    t1 = time.time()
    img = Image.open(io.BytesIO(img_data))

    # Convert to grayscale + sharpen for better OCR
    img = img.convert("L")
    img = img.filter(ImageFilter.SHARPEN)

    # Get detailed data with confidence scores
    data = pytesseract.image_to_data(img, lang="eng", config="--psm 6 --oem 3", output_type=pytesseract.Output.DICT)

    # Filter by confidence (Tesseract confidence 0-100)
    min_conf = 30
    valid_lines = []
    current_line = []
    current_conf = []

    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else 0
        line_num = data["line_num"][i]

        if conf >= min_conf and len(text) > 1 and text.isalpha():
            current_line.append(text)
            current_conf.append(conf)

    # Group by line_num from the raw data
    prev_line = -1
    line_texts = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else 0
        line_num = data["line_num"][i]

        if conf >= min_conf and len(text) > 1:
            if line_num != prev_line and line_texts:
                line_texts.append(" ".join(line_texts.pop()))
            line_texts.append(text)
            prev_line = line_num

    if not line_texts:
        log.info(f"OCR done in {time.time()-t1:.1f}s — no text found (all below confidence threshold)")
        return ""

    result = "\n".join(line_texts).strip()

    # Final quality check
    if not has_real_text(result):
        log.info(f"OCR done in {time.time()-t1:.1f}s — text discarded (low quality: {len(result)} chars)")
        return ""

    log.info(f"OCR done in {time.time()-t1:.1f}s — {len(result)} chars, {len(line_texts)} lines")
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python3 ocr.py <image_url>"}))
        sys.exit(1)
    try:
        text = ocr_image(sys.argv[1])
        print(json.dumps({"text": text}))
    except Exception as e:
        log.error(f"Failed: {e}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
