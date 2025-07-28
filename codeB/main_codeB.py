import os
import json
import re
import statistics
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from pdf2image import convert_from_path
from your_1a_code import extract_pdf_spans, classify_headers, group_adjacent_spans
import spacy
import unicodedata
import shutil

# ✅ Load spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")

# ✅ Detect ONNX model path (Docker vs Colab)
if os.path.exists("/app/onnx_model"):
    MODEL_DIR = "/app/onnx_model"
elif os.path.exists("/content/onnx_model"):
    MODEL_DIR = "/content/onnx_model"
else:
    raise FileNotFoundError("❌ Could not find onnx_model folder in /app or /content")

print(f"✅ Using ONNX model from: {MODEL_DIR}")

# ✅ Text normalization helper
def normalize_text(text):
    replacements = {
        "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2022": "-", "\u00a0": " ",
        "\u2212": "-", "\u2044": "/"
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return unicodedata.normalize("NFKC", text)

# ✅ Extract nouns, verbs, etc. for persona keywords
def extract_pos_info(role: str, task: str):
    text = f"{role}. {task}"
    doc = nlp(text)
    pos_tokens = {
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "ADJ", "VERB", "PROPN"}
           and token.is_alpha and not token.is_stop and len(token) > 3
    }
    entities = {
        ent.text.lower()
        for ent in doc.ents if len(ent.text) > 2
    }
    return list(pos_tokens | entities)

# ✅ Extract boost terms from persona
def extract_boost_terms(text):
    stopwords = {
        "the", "and", "or", "to", "for", "a", "of", "in", "on", "at", "with",
        "is", "as", "by", "an", "be", "will", "this", "that", "from"
    }
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if len(w) > 3 and w not in stopwords]

# ✅ PersonaRanker using ONNX model (patched with mean pooling)
class PersonaRanker:
    def __init__(self, role, job):
        self.persona_query = f"{role}. {job}"

        # ✅ Load tokenizer & ONNX model from detected MODEL_DIR
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        self.session = ort.InferenceSession(str(MODEL_DIR + "/model.onnx"))

        # ✅ Encode persona query ONCE
        self.query_embedding = self.encode(self.persona_query)
        self.boost_terms = extract_boost_terms(self.persona_query)
        print(f"🔍 Persona-driven boost terms: {self.boost_terms}")

    def encode(self, text):
        """Generate embeddings using ONNX model and pool token embeddings."""
        inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)

        # ✅ Only keep inputs that ONNX expects
        accepted_inputs = {inp.name for inp in self.session.get_inputs()}
        ort_inputs = {k: v for k, v in inputs.items() if k in accepted_inputs}

        ort_outs = self.session.run(None, ort_inputs)
        token_embeddings = ort_outs[0]  # shape: (1, seq_len, hidden_dim)

        # ✅ Mean Pooling over tokens (convert to (1, hidden_dim))
        sentence_embedding = token_embeddings.mean(axis=1)

        return sentence_embedding

    def cosine_similarity(self, a, b):
        a = a.flatten()
        b = b.flatten()
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def score_section(self, title, content):
        combined = f"{title}. {content}"
        section_embedding = self.encode(combined)
        base_score = self.cosine_similarity(self.query_embedding, section_embedding)
        boost = sum(1 for word in self.boost_terms if word in content.lower()) * 0.01
        if ":" in title and len(title.split()) > 5:
            base_score -= 0.02
        return float(base_score + boost)

# ✅ Match headings for each paragraph
def match_heading(span_group, headers):
    group_top = span_group[0]['y_top']
    group_page = span_group[0]['page']
    same_page = [h for h in headers if h['page'] == group_page]
    above = [h for h in same_page if h.get('y_top', 0) < group_top]
    if not above:
        return "General"
    return sorted(above, key=lambda h: group_top - h.get('y_top', 0))[0]['text']

# ✅ Extract sections for ranking
def extract_sections_for_ranking(pdf_path, spans, images, max_size_th, mid_size, max_size):
    headers = classify_headers(spans, images, max_size_th, mid_size, max_size)
    flat_spans = [s for page in spans for s in page]
    paragraphs = group_adjacent_spans(flat_spans)

    sections = []
    for para in paragraphs:
        text = normalize_text(" ".join([s["text"] for s in para]))
        if 15 < len(text) < 1500:
            section_title = match_heading(para, headers)
            if section_title.startswith(("•", "-", "–")) or len(section_title.split()) > 10:
                section_title = text.split(".")[0][:80]
            sections.append({
                "document": os.path.basename(pdf_path),
                "section_title": section_title,
                "content": text,
                "page": para[0]["page"]
            })
    return sections

# ✅ Pick clean title for document summary
def pick_clean_title(top_sections, doc_name):
    for sec in top_sections:
        title = sec["section_title"].strip()
        if not (title.startswith(("•", "-", "–")) or len(title.split()) > 10):
            return title
    return f"Summary of {doc_name}"

# ✅ MAIN function for generating JSON output
def generate_round1b_output(input_json_path, sections_to_merge=5):
    top_pdf_count = 5
    with open(input_json_path) as f:
        config = json.load(f)

    role = config["persona"]["role"]
    task = config["job_to_be_done"]["task"]
    documents = config["documents"]

    ranker = PersonaRanker(role, task)
    keyword_list = extract_pos_info(role, task)
    print(f"🧠 Extracted keywords & facts: {keyword_list}")

    pdf_section_info = []
    all_matched_sections = []

    for doc in documents:
        filename = doc["filename"]
        pdf_path = os.path.join("/app/input", filename)

        print(f"📄 Processing {filename}...")

        spans, threshold_size, doc_obj = extract_pdf_spans(pdf_path)
        images = convert_from_path(pdf_path, dpi=200)

        all_sizes = sorted({s["font_size"] for page in spans for s in page}, reverse=True)
        max_size_th = all_sizes[1] if len(all_sizes) > 1 else threshold_size
        mid_size = statistics.median(all_sizes) if len(all_sizes) > 1 else max_size_th * 0.85
        max_size = all_sizes[0] if all_sizes else threshold_size

        all_sections = extract_sections_for_ranking(pdf_path, spans, images, max_size_th, mid_size, max_size)
        print(f"   ↳ Found {len(all_sections)} total sections")

        matched_sections = []
        for sec in all_sections:
            if any(k in sec["content"].lower() for k in keyword_list):
                sec["score"] = ranker.score_section(sec["section_title"], sec["content"])
                matched_sections.append(sec)

        matched_sections = [
            s for s in matched_sections
            if s["section_title"].strip().lower() not in {"general", "conclusion"}
        ]

        match_count = len(matched_sections)
        print(f"   ↳ {match_count} keyword-matched sections")

        if match_count > 0:
            pdf_section_info.append({
                "filename": filename,
                "match_count": match_count,
                "sections": matched_sections
            })
            all_matched_sections.extend(matched_sections)

    # ✅ Pick top PDFs
    top_pdfs = sorted(pdf_section_info, key=lambda x: x["match_count"], reverse=True)[:top_pdf_count]

    # ✅ Pick best sections
    top_sections = []
    for pdf in top_pdfs:
        best_section = sorted(pdf["sections"], key=lambda x: x["score"], reverse=True)[0]
        best_section["section_title"] = pick_clean_title(pdf["sections"], pdf["filename"])
        top_sections.append(best_section)

    top_sections = sorted(top_sections, key=lambda x: x["score"], reverse=True)
    for i, sec in enumerate(top_sections):
        sec["importance_rank"] = i + 1

    # ✅ Pick top subsections
    subsection_limit = 5
    max_per_pdf = 2
    selected_subsections = []
    per_pdf_counter = Counter()
    for sec in sorted(all_matched_sections, key=lambda x: x["score"], reverse=True):
        doc = sec["document"]
        if per_pdf_counter[doc] < max_per_pdf:
            selected_subsections.append(sec)
            per_pdf_counter[doc] += 1
        if len(selected_subsections) >= subsection_limit:
            break

    # ✅ Final output JSON
    output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in documents],
            "persona": role,
            "job_to_be_done": task,
            "processing_timestamp": datetime.utcnow().isoformat()
        },
        "extracted_sections": [
            {
                "document": sec["document"],
                "section_title": sec["section_title"],
                "importance_rank": sec["importance_rank"],
                "page_number": sec["page"]
            } for sec in top_sections
        ],
        "subsection_analysis": [
            {
                "document": sec["document"],
                "refined_text": sec["content"],
                "page_number": sec["page"]
            } for sec in selected_subsections
        ]
    }

    with open("challenge1b_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Final Output: 1 top section from top 5 PDFs, and top 5 subsections overall.")
    print("✅ challenge1b_output.json saved.")

# ✅ ENTRYPOINT for Docker
if __name__ == "__main__":
    input_path = "/app/input/challenge1b_input.json"
    output_path = "/app/output/challenge1b_output.json"

    generate_round1b_output(input_path, sections_to_merge=5)

    # Move generated file to /app/output so it shows up in host output folder
    shutil.move("challenge1b_output.json", output_path)
    print(f"\n✅ Output saved to {output_path}")
