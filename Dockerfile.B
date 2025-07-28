FROM python:3.11-slim

WORKDIR /app

COPY . /app
COPY onnx_model /app/onnx_model   

# ✅ Install system packages
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# ✅ Install Python dependencies (WITHOUT NumPy first)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        spacy \
        transformers==4.41.2 \
        onnxruntime==1.18.0 \
        pdf2image \
        pillow \
        PyMuPDF \
        pydantic \
        sentencepiece \
        pytesseract

# ✅ Force-remove NumPy 2.x (if any package installed it)
RUN pip uninstall -y numpy && pip install --no-cache-dir numpy==1.26.4

# ✅ Download SpaCy model
RUN python -m spacy download en_core_web_sm

VOLUME ["/app/input", "/app/output"]

CMD ["python", "codeB/main_codeB.py"]
