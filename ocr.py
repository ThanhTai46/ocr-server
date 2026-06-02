#!/usr/bin/env python3
"""OCR Service — extracts text from images using Tesseract.
Filters out low-confidence, garbage, and merges spaced letters.
Usage: python3 ocr.py <image_url>
Returns: {"text": "extracted text"}
"""
import sys, json, urllib.request, io, time, logging, re
import pytesseract
from PIL import Image, ImageFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [OCR] %(levelname)s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
log = logging.getLogger("ocr")

def clean_text(text: str) -> str:
    """Merge spaced letters like 'b u n n y' → 'bunny' and remove garbage lines."""
    # Merge spaced single letters
    cleaned = re.sub(r'\b([a-zA-Z]) ([a-zA-Z])\b', r'\1\2', text)
    # Remove lines with mostly special characters
    lines = cleaned.split("\n")
    good = []
    for line in lines:
        line = line.strip()
        if not line: continue
        alnum = sum(1 for c in line if c.isalnum() or c.isspace())
        special = sum(1 for c in line if not c.isalnum() and not c.isspace())
        if special > alnum * 0.3 and len(line) > 5:
            continue
        good.append(line)
    return "\n".join(good)

def has_real_text(text: str) -> bool:
    text = clean_text(text)
    if not text or len(text) < 10:
        return False
    letters = sum(1 for c in text if c.isalpha())
    if letters / max(len(text), 1) < 0.5:
        return False
    alnum_chars = re.sub(r'[^a-zA-Z0-9]', '', text)
    unique_chars = len(set(alnum_chars.lower()))
    if unique_chars <= 3 and len(alnum_chars) > 10:
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
        log.info(f"OCR done in {time.time()-t1:.1f}s — no text found")
        return ""

    result = "\n".join(line_texts).strip()
    if not has_real_text(result):
        log.info(f"OCR done in {time.time()-t1:.1f}s — discarded (low quality: {len(result)} chars)")
        return ""

    result = clean_text(result)
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
