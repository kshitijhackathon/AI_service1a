# Multilingual PDF Analyzer for Adobe Hackathon Round 1B

## Overview
This project is a robust, multilingual PDF analyzer designed for the Adobe Hackathon Round 1B. It extracts structured outlines (titles, headings) from PDFs, semantically matches them to a persona/job description, and produces ranked, insight-rich JSON outputs for downstream automation or review.

## Folder Structure
```
app/
├── input/                # Place your input PDF files here
├── output/               # Extractor and analyzer outputs are saved here
├── extractor.py          # Extracts title and outline (H1/H2/H3) from PDFs
├── README.md             # This file
```

## How It Works
1. **extractor.py**
   - Extracts the document title and outline (H1/H2/H3) from each PDF in `/input`.
   - Handles multilingual content and noisy layouts.
   - Outputs a JSON per PDF in `/output` (e.g., `file01.json`).

## Usage
1. **Extract Outlines**
   - Place your PDFs in the `input/` folder.
   - Run:
     ```bash
     python extractor.py
     ```
   - This will generate one JSON per PDF in `output/`.



## Requirements
- Python 3.8+
- PyMuPDF (fitz)
- pdf2image
- pytesseract
- Pillow
- numpy
- sentence-transformers (all-MiniLM-L6-v2)
- scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Features
- Multilingual OCR and heading detection (supports English, Hindi, Chinese, Japanese, Arabic, Russian, and more)
- Semantic section ranking using embeddings
- Robust to noisy, scanned, or digital PDFs
- Modular and extensible for future enhancements

## Notes
- All processing is CPU-only and offline (no internet required)
- Model size and runtime are within hackathon constraints
- Exception handling for missing/invalid files is included

## Contact
For questions or issues, contact the project owner or submit an issue in your repository.
# AI_service1b
