Here’s a **polished and judge‑friendly README.md** you can drop into your repo:

---

```markdown
# 🏆 1B – Adobe Hackathon Round 1B Solution

This repository contains the **Dockerized solution** for Round 1B of the Adobe Hackathon.  
It processes PDF documents based on a provided persona and task, then outputs structured JSON.

---

## 📂 Project Structure

```

myproject/
│── codeB/                # Main code for 1B
│   ├── main\_codeB.py      # Entry point (Round 1B logic)
│   ├── your\_1A\_code.py    # Helper code reused from Round 1A
│── onnx\_model/            # Tokenizer + ONNX model files (Git LFS)
│── input/                 # Judges will place PDFs + input JSON here
│── output/                # Generated JSON output will appear here
│── Dockerfile.B           # Dockerfile for building this solution
│── README.md              # This file

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
docker build -f Dockerfile.B -t codeb:latest .
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
docker run --rm \
   -v ${PWD}/input:/app/input \
   -v ${PWD}/output:/app/output \
   codeb:latest
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

## 🛠 Requirements

* 🐳 Docker (latest version)
* 🖥️ At least **4GB RAM** recommended for smooth PDF processing

---

## ✅ Done! 🎉

This setup ensures **zero manual hassle** – the system works out‑of‑the‑box for any PDF & persona combination.


