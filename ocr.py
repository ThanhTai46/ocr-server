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

COMMON_WORDS = {"a", "an", "in", "on", "at", "to", "is", "it", "of", "by", "be",
                 "he", "she", "we", "they", "me", "my", "no", "so", "go", "up",
                 "us", "or", "as", "do", "if", "am", "ex", "oh", "ok", "tv",
                 "was", "had", "has", "but", "can", "for", "the", "and", "are",
                 "not", "his", "her", "its", "all", "how", "why", "who", "did",
                 "get", "got", "out", "say", "see", "too", "two", "way", "now",
                 "may", "man", "say", "let", "him"}

def clean_text(text: str) -> str:
    """Merge spaced letters like 'b u n n y' → 'bunny' and remove garbage."""
    before = text

    # Merge spaced single letters across multiple chars: "t h e" → "the"
    cleaned = re.sub(r'(?<![a-zA-Z])([a-zA-Z]) (?:([a-zA-Z]) )+([a-zA-Z])(?![a-zA-Z])',
                     lambda m: m.group(0).replace(" ", ""), text)
    # Also handle pairs: "a b" → "ab" (if within a longer word context)
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

        # Drop lines that are mostly special characters
        if special > max(letters, 1) * 0.6 and total > 2:
            log.info(f"  clean: dropped garbage ({total}c, {special}sp): {line[:60]}")
            continue

        # Keep lines with 4+ alphabetic chars (strip punctuation for count)
        word_letters = len(re.sub(r'[^a-zA-Z]', '', line))
        if word_letters >= 4:
            good.append(line)
            continue

        # Keep common short words
        word = re.sub(r'[^a-zA-Z]', '', line).lower()
        if word in COMMON_WORDS:
            good.append(line)
            continue

        # Keep short lines that are mostly alphabetic
        if letters >= 3 and word_letters == len(line.rstrip(".,!?;:\"'")):
            good.append(line)
            continue

        # Log what was dropped
        if total > 2:
            log.info(f"  clean: dropped noise ({total}c, {letters}alpha): {line[:60]}")

    result = "\n".join(good)
    if result != before:
        log.info(f"  clean: removed garbage / merged letters ({len(before)}→{len(result)} chars)")
    return result

def has_real_text(text: str) -> bool:
    text = clean_text(text)
    if not text or len(text) < 10:
        log.info(f"  validate: too short ({len(text) if text else 0} chars)")
        return False
    letters = sum(1 for c in text if c.isalpha())
    letter_ratio = letters / max(len(text), 1)
    if letter_ratio < 0.5:
        log.info(f"  validate: low letter ratio ({letter_ratio:.0%})")
        return False
    alnum_chars = re.sub(r'[^a-zA-Z0-9]', '', text)
    unique_chars = len(set(alnum_chars.lower()))
    if unique_chars <= 3 and len(alnum_chars) > 10:
        log.info(f"  validate: too few unique chars ({unique_chars})")
        return False
    words = re.findall(r"[a-zA-Z]{4,}", text)
    if len(words) >= 1:
        return True
    words3 = re.findall(r"[a-zA-Z]{3,}", text)
    if len(words3) >= 3:
        return True
    log.info(f"  validate: not enough real words ({len(words3)} words ≥3 chars)")
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
    log.info(f"Image size: {img.size}, mode: {img.mode}")

    img = img.convert("L")
    log.info("Converted to grayscale")
    img = img.filter(ImageFilter.SHARPEN)
    log.info("Applied sharpen filter")

    log.info("Running Tesseract OCR (psm 6, oem 3)...")
    data = pytesseract.image_to_data(img, lang="eng", config="--psm 6 --oem 3", output_type=pytesseract.Output.DICT)

    total_blocks = len(data["text"])
    valid_blocks = sum(1 for t, c in zip(data["text"], data["conf"]) if t.strip() and int(c) >= 20)
    log.info(f"Tesseract: {total_blocks} blocks, {valid_blocks} with conf≥20")

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
        elif conf > 0 and conf < 20 and len(text) > 1:
            log.info(f"  low-conf block ({conf}): {text[:50]}")

    if not line_texts:
        log.info(f"OCR done in {time.time()-t1:.1f}s — no text found")
        return ""

    result = "\n".join(line_texts).strip()
    log.info(f"Raw OCR: {len(line_texts)} lines, {len(result)} chars")
    log.info(f"--- RAW TEXT ---\n{result[:500]}\n---")

    if not has_real_text(result):
        log.info(f"OCR done in {time.time()-t1:.1f}s — discarded (low quality: {len(result)} chars)")
        return ""

    result = clean_text(result)
    log.info(f"Final text: {len(result)} chars, {len(line_texts)} lines")
    log.info(f"--- FINAL TEXT ---\n{result[:500]}\n---")
    log.info(f"OCR done in {time.time()-t1:.1f}s")
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
