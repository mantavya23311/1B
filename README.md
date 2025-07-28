# 1B
## Solution code for 1B

## Name of team members:
1) Mantavya
2) Arnav
3) Anshul

## Dependencies for the 1B_solution code:
1) import os
2) import json
3) import re
4) import statistics
5) from datetime import datetime
6) from collections import defaultdict, Counter
7) import numpy as np
8) import onnxruntime as ort
9) from transformers import AutoTokenizer
10) from pdf2image import convert_from_path
11) from your_1a_code import extract_pdf_spans, classify_headers, group_adjacent_spans
12) import spacy
13) import unicodedata
14) import shutil

## This also support Multilingual pdfs summarizer:
The solution for 1B also support multilingual pdf for summarizing and extracting relevent headers from them, as it is build upon 1A solution which is written in your_1a_code file.
---
```markdown
# 🏆 1B – Adobe Hackathon Round 1B Solution

This repository contains the **Dockerized solution** for Round 1B of the Adobe Hackathon.  
It processes PDF documents based on a provided persona and task, then outputs structured JSON.

---
```

## 📂 Project Structure
<img width="1280" height="226" alt="image" src="https://github.com/user-attachments/assets/abc7f4ab-a4da-4230-9859-bd4e5604e501" />

````

---

## 🚀 How to Run

### ✅ 1. **Clone the Repository**
```bash
git clone https://github.com/mantavya23311/1B.git
cd 1B
````

---

### ✅ 2. **Install Docker**

Make sure Docker Desktop is installed and running.
👉 [Download Docker](https://www.docker.com/products/docker-desktop/)

---

### ✅ 3. **Build the Docker Image**

Run this inside the project folder:

```bash
docker build --platform linux/amd64 -t codeb:latest -f Dockerfile.B .

```

---

### ✅ 4. **Prepare Input Files**

1️⃣ Place all **PDF files** in the `input/` folder.
2️⃣ Add a **`challenge1b_input.json`** file in the same folder.

Example `challenge1b_input.json`:

```json
{
  "persona": { "role": "HR Manager" },
  "job_to_be_done": { "task": "create and manage fillable forms for onboarding and compliance" },
  "documents": [
    { "filename": "Learn Acrobat - Create and Convert_1.pdf" },
    { "filename": "Learn Acrobat - Edit_1.pdf" }
  ]
}
```

---

### ✅ 5. **Run the Container**

```bash
  docker run --rm -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output codeb:latest


```
---

### ✅ 6. **Check the Output**

* After the container finishes, check the `output/` folder.
* You’ll see:

  * ✅ `challenge1b_output.json` → Final processed output.

---

## 📥 Input & 📤 Output Explanation

* **📥 Input Folder** (`input/`)

  * Judges place PDFs & `challenge1b_input.json` here.

* **📤 Output Folder** (`output/`)

  * The script writes the **final JSON** here automatically.

---

## 🔍 Notes for Judges

✔ No manual code edits are needed.
✔ Just add PDFs & JSON → run Docker → get JSON output.
✔ The ONNX model is already included via **Git LFS**, so no extra setup required.

---
---
Here’s a **clear and judge‑friendly Deliverables section** you can drop into your `README.md` under the **bottom of the file**:

---

## 📦 Deliverables

Our Round 1B submission includes everything required for judges to execute, verify, and evaluate the solution without extra setup.

### ✅ What’s Included

* **Full Codebase** – Organized in `codeB/`, containing:

  * `main_codeB.py` (the **entry point** running all Round 1B logic).
  * `your_1A_code.py` (helper code reused from Round 1A for span extraction, header detection, and grouping).
* **Dockerfile.B** – A complete Docker configuration that sets up **all dependencies** automatically (spacy, transformers, onnxruntime, pdf2image, etc.).
* **ONNX Model** – Provided via **Git LFS** inside `onnx_model/` so judges don’t have to download or configure models manually.
* **Sample Project Structure Image** – A visual representation of how the repository is laid out for clarity.
* **Input & Output Folders** – Pre‑made folders for judges:

  * `/input` → Place PDFs and a `challenge1b_input.json`.
  * `/output` → The processed `challenge1b_output.json` will be generated here.

### ✅ How It Executes

1. Judges simply **build the Docker image**:

   ```bash
   docker build --platform linux/amd64 -t codeb:latest -f Dockerfile.B .
   ```

   🛠 This step automatically installs all system libraries (Tesseract OCR, Poppler, etc.) and Python dependencies, meaning **no manual pip installs** are needed.

2. Judges **place PDFs & input JSON** in `/input` and run the container:

   ```bash
   docker run --rm -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output codeb:latest
   ```

   ✅ The container scans the PDFs, extracts sections, ranks them via persona‑aware embeddings (ONNX), and outputs the final JSON to `/output`.

3. **Output is fully self‑contained** – Judges will find a single `challenge1b_output.json` summarizing top sections and subsections, ready for review.

### ✅ Why This is Judge‑Friendly

* **Zero local setup**: No need to install Python, transformers, or OCR tools – Docker handles it all.
* **Reusable & Flexible**: Judges can swap in *any* PDFs and persona JSON without code changes.
* **Lightweight execution**: The ONNX model ensures **fast inference** without requiring GPU, PyTorch, or TensorFlow.
* **Transparent folder structure**: Clear input → processing → output flow, fully isolated in Docker.

This guarantees that the judging team can **clone, build, run, and verify** the solution in minutes, focusing on evaluation instead of troubleshooting.

---




---



## ✅ Done! 🎉

This setup ensures **zero manual hassle** – the system works out‑of‑the‑box for any PDF & persona combination.


