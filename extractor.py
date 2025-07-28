import os
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re
from collections import defaultdict
from multiprocessing import Pool

# Core PDF and OCR libraries
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFOutlineExtractor:
    """
    Intelligent PDF outline extractor supporting multilingual documents
    """
    
    def __init__(self):
        # Multilingual OCR configuration
        self.ocr_languages = "eng+hin+tam+tel+ben+guj+kan+mar+jpn"
        self.ocr_config = "--psm 6 -c tessedit_do_invert=0"
        
        # Heading detection thresholds (height in pixels)
        self.h1_min_height = 30
        self.h2_min_height = 18
        self.h3_min_height = 12
        
        # Title detection parameters
        self.title_min_height = 50
        self.center_tolerance = 0.3
        
        # OCR confidence threshold
        self.min_confidence = 60
        
        # Performance settings
        self.dpi = 300
        
    def load_pdf(self, pdf_path: str) -> fitz.Document:
        """Load PDF document with error handling"""
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"Loaded PDF: {pdf_path} ({len(doc)} pages)")
            return doc
        except Exception as e:
            logger.error(f"Failed to load PDF {pdf_path}: {e}")
            raise
    
    def extract_embedded_text(self, doc: fitz.Document) -> Dict[int, List[Dict]]:
        """Extract embedded text with positioning info from digital PDFs"""
        text_blocks = {}
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            blocks = []
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_bbox = None
                        font_size = 0
                        
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            if text:
                                line_text += text + " "
                                if line_bbox is None:
                                    line_bbox = span["bbox"]
                                else:
                                    line_bbox = [
                                        min(line_bbox[0], span["bbox"][0]),
                                        min(line_bbox[1], span["bbox"][1]),
                                        max(line_bbox[2], span["bbox"][2]),
                                        max(line_bbox[3], span["bbox"][3])
                                    ]
                                font_size = max(font_size, span.get("size", 0))
                        
                        if line_text.strip() and line_bbox:
                            blocks.append({
                                "text": line_text.strip(),
                                "bbox": line_bbox,
                                "height": font_size,
                                "page": page_num + 1,
                                "confidence": 100,
                                "source": "embedded"
                            })
            
            text_blocks[page_num + 1] = blocks
            
        return text_blocks
    
    def convert_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to high-resolution images for OCR"""
        try:
            images = convert_from_path(
                pdf_path, 
                dpi=self.dpi,
                fmt='RGB',
                thread_count=1
            )
            logger.info(f"Converted {len(images)} pages to images")
            return images
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return []
    
    @staticmethod
    def perform_ocr(image: Image.Image, page_num: int, crop_top: bool, ocr_languages: str, ocr_config: str, min_confidence: int) -> List[Dict]:
        """Perform OCR with detailed bounding box information"""
        try:
            if crop_top:
                width, height = image.size
                crop_height = int(height * 0.3)
                image = image.crop((0, 0, width, crop_height))
            
            img_array = np.array(image)
            ocr_data = pytesseract.image_to_data(
                img_array,
                lang=ocr_languages,
                config=ocr_config,
                output_type=Output.DICT
            )
            
            lines = PDFOutlineExtractor._group_words_into_lines(ocr_data, page_num, min_confidence)
            return lines
            
        except Exception as e:
            print(f"OCR failed for page {page_num}: {e}")
            return []
    
    @staticmethod
    def _group_words_into_lines(ocr_data: Dict, page_num: int, min_confidence: int) -> List[Dict]:
        """Group OCR words into lines and estimate font sizes based on bounding boxes"""
        lines = []
        line_groups = defaultdict(list)
        
        for i in range(len(ocr_data['text'])):
            conf = int(ocr_data['conf'][i])
            text = ocr_data['text'][i].strip()
            
            if conf >= min_confidence and text:
                line_num = ocr_data['line_num'][i]
                line_groups[line_num].append({
                    'text': text,
                    'left': ocr_data['left'][i],
                    'top': ocr_data['top'][i],
                    'width': ocr_data['width'][i],
                    'height': ocr_data['height'][i],
                    'conf': conf
                })
        
        for line_num, words in line_groups.items():
            if not words:
                continue
                
            words.sort(key=lambda x: x['left'])
            line_text = ' '.join(word['text'] for word in words)
            left = min(word['left'] for word in words)
            top = min(word['top'] for word in words)
            right = max(word['left'] + word['width'] for word in words)
            bottom = max(word['top'] + word['height'] for word in words)
            
            height = bottom - top
            avg_conf = sum(word['conf'] for word in words) / len(words)
            
            lines.append({
                'text': line_text,
                'bbox': [left, top, right, bottom],
                'height': height,
                'page': page_num,
                'confidence': avg_conf,
                'source': 'ocr'
            })
        
        return lines
    
    def detect_title(self, text_blocks: Dict[int, List[Dict]]) -> Optional[str]:
        """Detect document title from first page using largest heading if no metadata title"""
        if 1 not in text_blocks:
            return None
        first_page_blocks = text_blocks[1]
        largest_block = None
        for block in first_page_blocks:
            text = block['text'].strip()
            height = block['height']
            conf = block['confidence']
            if conf >= self.min_confidence and len(text.split()) >= 2 and len(text) <= 200:
                if largest_block is None or height > largest_block['height']:
                    largest_block = block
        if largest_block:
            title = re.sub(r'\s+', ' ', largest_block['text']).strip()
            if (re.match(r'^rsvp[:\-\s]*$', title, re.IGNORECASE) or
                re.match(r'^[\W_]+$', title) or
                len(re.sub(r'[^A-Za-z]', '', title)) < 3):
                logger.info(f"Filtered out non-meaningful title: {title}")
                return ""
            logger.info(f"Detected title: {title}")
            return title
        return ""
    
    def detect_headings(self, text_blocks: Dict[int, List[Dict]], title: Optional[str] = None) -> List[Dict]:
        """Detect and classify headings, excluding the title and non-meaningful headings"""
        headings = []
        seen_texts = set()
        for page_num, blocks in text_blocks.items():
            for block in blocks:
                text = block['text'].strip()
                height = block['height']
                conf = block['confidence']
                if (text in seen_texts or conf < self.min_confidence or
                    (re.match(r'^rsvp[:\-\s]*$', text, re.IGNORECASE)) or
                    re.match(r'^[\W_]+$', text) or
                    len(re.sub(r'[^A-Za-z]', '', text)) < 3):
                    continue
                if self._is_heading_candidate(text, height, block):
                    if title and text == title:
                        continue
                    level = self._classify_heading_level(height)
                    heading = {
                        'level': level,
                        'text': text,
                        'page': page_num - 1,
                        'height': height,
                        'confidence': conf
                    }
                    headings.append(heading)
                    seen_texts.add(text)
        headings.sort(key=lambda x: (x['page'], -x['height']))
        return headings
    
    def _is_heading_candidate(self, text: str, height: float, block: Dict) -> bool:
        """Determine if text is likely a heading"""
        if height < self.h3_min_height:
            return False
        word_count = len(text.split())
        if word_count < 1 or word_count > 20:
            return False
        if len(text) < 2 or len(text) > 150:
            return False
        if re.match(r'^[\d\s\.\-]+$', text):
            return False
        text_lower = text.lower()
        skip_patterns = [
            r'^page \d+',
            r'^\d+$',
            r'^figure \d+',
            r'^table \d+',
            r'^appendix',
            r'^references?$',
            r'^bibliography$'
        ]
        for pattern in skip_patterns:
            if re.match(pattern, text_lower):
                return False
        if len(re.sub(r'[^A-Za-z]', '', text)) < 2:
            return False
        return True
    
    def _classify_heading_level(self, height: float) -> str:
        """Classify heading level based on font size"""
        if height >= self.h1_min_height:
            return "H1"
        elif height >= self.h2_min_height:
            return "H2"
        else:
            return "H3"
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Main processing pipeline for a single PDF"""
        logger.info(f"Processing PDF: {pdf_path}")
        try:
            doc = self.load_pdf(pdf_path)
            text_blocks = self.extract_embedded_text(doc)
            total_embedded_text = sum(
                len(' '.join(block['text'] for block in blocks))
                for blocks in text_blocks.values()
            )
            if total_embedded_text < 100:
                logger.info("Minimal embedded text found, performing OCR...")
                doc.close()
                images = self.convert_to_images(pdf_path)
                with Pool(processes=8) as pool:
                    text_blocks_list = pool.starmap(
                        PDFOutlineExtractor.perform_ocr,
                        [(img, i+1, i>0, self.ocr_languages, self.ocr_config, self.min_confidence) for i, img in enumerate(images)]
                    )
                text_blocks = {i+1: blocks for i, blocks in enumerate(text_blocks_list)}
            else:
                logger.info(f"Using embedded text ({total_embedded_text} characters)")
                doc.close()
            title = self.detect_title(text_blocks)
            headings = self.detect_headings(text_blocks, title)
            outline = []
            first_page_headings = [h for h in headings if h["page"] == 0]
            if first_page_headings:
                filtered = [h for h in first_page_headings if title is None or h["text"].strip() != title.strip()]
                if filtered:
                    largest = max(filtered, key=lambda h: h["height"])
                    outline.append({
                        "level": "H1",
                        "text": largest["text"],
                        "page": largest["page"]
                    })
            result = {
                "title": title if title is not None else "",
                "outline": outline
            }
            logger.info(f"Extracted title: {result['title']}")
            logger.info(f"Found {len(result['outline'])} headings")
            return result
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            logger.error(traceback.format_exc())
            return {
                "title": "Error Processing Document",
                "outline": []
            }
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all PDFs in input directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(input_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                result = self.process_pdf(str(pdf_file))
                output_file = output_path / f"{pdf_file.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved result to {output_file}")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                error_result = {
                    "title": "Processing Error",
                    "outline": []
                }
                output_file = output_path / f"{pdf_file.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(error_result, f, indent=2, ensure_ascii=False)

def main():
    """Main entry point"""
    logger.info("Starting PDF Outline Extractor")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    extractor = PDFOutlineExtractor()
    extractor.process_directory(input_dir, output_dir)
    logger.info("Processing complete")

if __name__ == "__main__":
    main()