# Use Python 3.10 slim image for optimal size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PDF processing and OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-tam \
    tesseract-ocr-tel \
    tesseract-ocr-ben \
    tesseract-ocr-guj \
    tesseract-ocr-kan \
    tesseract-ocr-mar \
    tesseract-ocr-jpn \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Tesseract optimization for Docker
ENV OMP__THREAD__LIMIT=1
ENV TESSDATA__PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY extractor.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set the entry point
CMD ["python", "extractor.py"]
