import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from PIL import Image
from collections import defaultdict
import json
import os
import re
import fitz
import statistics
from concurrent.futures import ThreadPoolExecutor



INPUT_DIR = "/app/input"
PDF_FILES = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

OUTPUT_JSON = "output.json"



# --- REGEX CACHES ---
RE_DIGIT = re.compile(r"^\d+\.?$")
RE_OPTION = re.compile(r"^\(?[a-zA-Z]\)?$")
RE_RS = re.compile(r'^[A-Za-z]{1,4}$')
RE_BULLETS = re.compile(r'[•●\-→]')
RE_LONG_TEXT = re.compile(r"[+@=#*~^$%&]")
RE_NUMBERED = re.compile(r"^\s*\d+\.")
# More comprehensive regex for dates, including month names, numbers, and common separators
RE_DATE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b"
    r"|\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b" # e.g., 03/21/2003, 21-03-2003
    r"|\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b(?:,\s*)?\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b"
    r"|\b\d{4}\s*-\s*\d{4}\b" # Year ranges like 2003-2004
)


# --- HELPERS ---
def normalize_box(box, image_size):
    w, h = image_size
    x0, y0, width, height = box
    return [
        int(1000 * x0 / w),
        int(1000 * y0 / h),
        int(1000 * (x0 + width) / w),
        int(1000 * (y0 + height) / h)
    ]

def is_inside(rect, container):
    return (rect.x0 >= container.x0 and rect.x1 <= container.x1 and
            rect.y0 >= container.y0 and rect.y1 <= container.y1)

def detect_table_regions(page):
    drawings = page.get_drawings()
    hlines, vlines = [], []

    for d in drawings:
        for x0, y0, x1, y1, _, _ in d["items"]:
            if abs(y1 - y0) < 1:
                hlines.append((x0, y0, x1, y1))
            elif abs(x1 - x0) < 1:
                vlines.append((x0, y0, x1, y1))

    table_boxes = []
    for h1 in hlines:
        for h2 in hlines:
            if h1 == h2: continue
            for v1 in vlines:
                for v2 in vlines:
                    if v1 == v2: continue
                    rect = fitz.Rect(min(v1[0], v2[0]), min(h1[1], h2[1]),
                                     max(v1[0], v2[0]), max(h1[1], h2[1]))
                    if rect.get_area() > 1000:
                        table_boxes.append(rect)
    return table_boxes

def detect_large_grouped_boxes(spans):
    grouped_boxes = []
    spans = sorted(spans, key=lambda x: (x["y_top"], x["x0"]))
    used = set()

    for i, span in enumerate(spans):
        if i in used:
            continue
        group = [span]
        for j in range(i + 1, len(spans)):
            s2 = spans[j]
            if abs(s2["y_top"] - span["y_top"]) < 50 and abs(s2["x0"] - span["x0"]) < 100:
                group.append(s2)
                used.add(j)
        if len(group) >= 2:
            xs = [g["x0"] for g in group] + [g["x1"] for g in group]
            ys = [g["y_top"] for g in group] + [g["y_bottom"] for g in group]
            rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
            grouped_boxes.append(rect)
    return grouped_boxes

def remove_table_and_grouped_box_spans(spans, page):
    table_boxes = detect_table_regions(page)
    grouped_boxes = detect_large_grouped_boxes(spans)
    boxes_to_ignore = table_boxes + grouped_boxes

    filtered = []
    for span in spans:
        rect = fitz.Rect(span["x0"], span["y_top"], span["x1"], span["y_bottom"])
        if not any(is_inside(rect, box) for box in boxes_to_ignore):
            filtered.append(span)
    return filtered

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_pdf_spans(pdf_path):
    doc = fitz.open(pdf_path)
    all_spans = []
    all_fonts = []

    for page_num, page in enumerate(doc, start=1):
        spans = []
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                line_spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not line_spans:
                    continue
                merged_spans = []
                i = 0
                while i < len(line_spans):
                    span = line_spans[i]
                    text = span["text"].strip()
                    merged_text = text
                    bbox = span["bbox"]
                    size = span["size"]
                    flags = span["flags"]
                    j = i + 1
                    while merged_text.endswith("-") and j < len(line_spans):
                        next_span = line_spans[j]
                        next_text = next_span["text"].strip()
                        if next_text:
                            merged_text = merged_text.rstrip("-") + "-" + next_text
                            bbox = (
                                bbox[0],
                                bbox[1],
                                next_span["bbox"][2],
                                max(bbox[3], next_span["bbox"][3])
                            )
                            size = max(size, next_span["size"])
                            flags |= next_span["flags"]
                        j += 1
                    merged_spans.append({
                        "text": merged_text,
                        "bbox": bbox,
                        "font_size": size,
                        "is_bold": (flags & 2) != 0,
                        "page": page_num,
                        "y_top": bbox[1],
                        "y_bottom": bbox[3],
                        "x0": bbox[0],
                        "x1": bbox[2],
                    })
                    all_fonts.append(size)
                    i = j
                spans.extend(merged_spans)
        all_spans.append(spans)

    global_threshold = sorted(all_fonts)[-6] if len(all_fonts) >= 6 else max(all_fonts, default=26)
    return all_spans, global_threshold, doc

def group_adjacent_spans(spans, y_thresh_same_line=5.0, y_thresh_block=25.0):
    if not spans:
        return []

    grouped = []
    current_group = [spans[0]]

    for i in range(1, len(spans)):
        prev = current_group[-1]
        curr = spans[i]

        same_line = abs(curr["y_top"] - prev["y_top"]) <= y_thresh_same_line
        similar_size = abs(curr["font_size"] - prev["font_size"]) <= 2
        same_bold = curr["is_bold"] == prev["is_bold"]
        vertical_proximity = (curr["y_top"] - prev["y_bottom"]) < y_thresh_block and (curr["y_top"] > prev["y_top"])

        if same_line or (similar_size and same_bold and vertical_proximity):
            current_group.append(curr)
        else:
            grouped.append(current_group)
            current_group = [curr]

    if current_group:
        grouped.append(current_group)

    return grouped

def infer_title(pages_spans, doc, max_size_th):
    candidates = []
    if pages_spans:
        page_num = 1
        spans = pages_spans[0] # Only consider the first page
        height = doc[page_num - 1].rect.height
        width = doc[page_num - 1].rect.width
        for span in spans:
            text = span["text"]
            y = span["y_top"]
            x_center = (span["x0"] + span["x1"]) / 2
            size = span["font_size"]
            # Exclude dates from title candidates
            # Exclude page numbers, dates, very short/long texts, and bullet points from title candidates
            if (RE_DIGIT.match(text) and len(text) <= 4) or RE_DATE.search(text) or RE_BULLETS.search(text):
                continue
            if y > height * 0.7 or len(text.split()) < 3 or len(text.split()) > 30:
                continue
            score = 0
            if span["is_bold"]: score += 5
            if size > max_size_th: score += 10
            elif size >= max_size_th * 1.2: score += 5 # Good size for title
            if text.istitle(): score += 2
            # Positional Scoring: Higher on the page gets more points
            if y < height * 0.15: # Very top 15%
                score += 8
            elif y < height * 0.30: # Top 30%
                score += 5
            elif y < height * 0.50: # Top 50%
                score += 2

            # Centering score (still important but now balanced with vertical preference)
            center_dist = abs(x_center - width / 2)
            score -= (center_dist / width) * 15 # Slightly reduced penalty to balance with vertical score

            candidates.append((score, size, y, text))
    if candidates:
        # Sort by score (desc), then size (desc), then y_top (asc)
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return candidates[0][3] # Return the text of the best candidate
    return "Untitled Document"

def is_potentially_heading(span, max_size_th, mid_size, max_size):
    text = span["text"].strip()
    size = span["font_size"]
    is_bold = span["is_bold"]

    if len(text) <= 3 and RE_DIGIT.match(text):
        return False
    if RE_OPTION.match(text):
        return False
    if RE_RS.fullmatch(text) and not is_bold:
        return False
    if text.endswith(":") or text.endswith("."):
        return False
    if RE_BULLETS.search(text):
        return False
    if len(text.split()) > 15:
        return False
    # New check: Exclude dates from being considered as headings
    if RE_DATE.search(text):
        return False
    if re.match(r'^\d+(\.\d+)*\s+\S+', text):
        if is_bold or size >= mid_size:
            return True
        return False
    if not is_bold and size <= max_size_th*1.1:
        return False
    return True

def classify_headers(pages_spans, images, max_size_th, mid_size ,max_size):
    all_headers = []

    for page_num, (spans, image) in enumerate(zip(pages_spans, images), start=1):
        if not spans:
            continue

        for span in spans:
            text = span["text"]
            size = span["font_size"]
            is_bold = span["is_bold"]
            y = span["y_top"]

            # First, check for bullet points and classify as H2
            if RE_BULLETS.match(text):
                # Remove the bullet point from the text for the header output
                cleaned_text = RE_BULLETS.sub('', text, 1).strip()
                # Ensure it's not just a bullet or very short, and not a long paragraph masquerading as a bulleted heading
                if cleaned_text and len(cleaned_text.split()) > 1 and len(cleaned_text.split()) <= 20:
                    all_headers.append({
                        "level": "H2", # Force H2 for bulleted items
                        "text": cleaned_text,
                        "page": page_num
                    })
                continue # Skip further classification for this span if it's a bullet

            if not is_potentially_heading(span, max_size_th, mid_size, max_size):
                continue
            if y >= image.size[1] * 0.9:
                continue
            if len(text.split()) > 15 or RE_LONG_TEXT.search(text):
                continue

            if size >= max_size_th:
                level = "H1"
            elif size >= mid_size-2:
                level = "H2"
            else:
                level = None

            if level and (is_bold or size >= mid_size-2):
                all_headers.append({
                    "level": level,
                    "text": text.strip(),
                    "page": page_num
                })

        grouped_blocks = group_adjacent_spans(spans)

        for group in grouped_blocks:
            numbered_count = sum(1 for s in group if RE_NUMBERED.match(s["text"]))
            if numbered_count >= 2:
                continue

            full_text = " ".join([
                s["text"].rstrip("-") + ("-" if s["text"].endswith("-") else "")
                for s in group
            ])
            full_text = full_text.replace("  ", " ").strip()

            # If this grouped block was already handled as a bullet point, skip
            if any(RE_BULLETS.match(s["text"].strip()) for s in group):
                continue

            # Check if it's a numbered list item (e.g., "1. Item")
            numbered_match = RE_NUMBERED.match(full_text)
            if numbered_match:
                # If it's a numbered item, classify as H2
                cleaned_text = RE_NUMBERED.sub('', full_text, 1).strip()
                if cleaned_text and len(cleaned_text.split()) > 1:
                    all_headers.append({
                        "level": "H2",
                        "text": cleaned_text,
                        "page": page_num
                    })
                continue # Skip further classification for this grouped block

            avg_size = max(s["font_size"] for s in group)
            is_bold = any(s["is_bold"] for s in group)
            y = group[0]["y_top"]

            if y >= image.size[1] * 0.9:
                continue
            if len(full_text.split()) > 15 or RE_LONG_TEXT.search(full_text):
                continue

            level = None
            if avg_size >= max_size_th:
                level = "H1"
            elif avg_size >= mid_size-2:
                level = "H2"

            if level and (is_bold or avg_size >= mid_size-2):
                all_headers.append({
                    "level": level,
                    "text": full_text,
                    "page": page_num
                })

    seen = set()
    unique = []
    for h in all_headers:
        key = (normalize_text(h["text"]), h["page"], h["level"])
        if key not in seen:
            seen.add(key)
            unique.append(h)

    return unique

def save_json(title, headers, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"title": title, "outline": headers}, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {out_file}")


# --- MAIN ---
def main():
    print("📂 Scanning input folder for PDFs...")
    if not PDF_FILES:
        print("⚠️ No PDF files found in /app/input")
        return

    for pdf_file in PDF_FILES:
        PDF_PATH = os.path.join(INPUT_DIR, pdf_file)
        print(f"📄 Processing {pdf_file}...")

        def load_spans():
            return extract_pdf_spans(PDF_PATH)

        def load_images():
            return convert_from_path(PDF_PATH, dpi=200)

        with ThreadPoolExecutor() as executor:
            future1 = executor.submit(load_spans)
            future2 = executor.submit(load_images)
            spans, threshold_size, doc = future1.result()
            all_sizes = sorted({span["font_size"] for page in spans for span in page}, reverse=True)
            print(f"   ↳ Detected font sizes: {all_sizes}")
            max_size_th = all_sizes[1] if all_sizes else threshold_size
            mid_size = statistics.median(all_sizes) if len(all_sizes) > 1 else max_size_th * 0.85
            max_size = all_sizes[0] if all_sizes else threshold_size
            images = future2.result()

        print("🧠 Inferring headers...")
        headers = classify_headers(spans, images, max_size_th, mid_size, max_size)

        print("🏷️ Inferring title...")
        title = infer_title(spans, doc, max_size_th)

        # ✅ Save each PDF’s output separately
        output_filename = os.path.splitext(pdf_file)[0] + ".json"
        output_path = os.path.join("/app/output", output_filename)

        print(f"💾 Writing {output_filename}...")
        save_json(title, headers, output_path)


if __name__ == "__main__":
    main()