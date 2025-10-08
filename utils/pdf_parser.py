"""
PDF Parser Module
Extracts text from PDF files with precise location tracking
"""

import fitz  # PyMuPDF
from typing import List, Dict, Tuple


def extract_text_with_location(pdf_path: str) -> List[Dict]:
    """
    Extract text from PDF with page numbers and bounding box coordinates
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing text, page number, and bbox coordinates
    """
    data = []
    
    try:
        with fitz.open(pdf_path) as doc:
            for page_no, page in enumerate(doc, start=1):
                # Get text blocks with position information
                blocks = page.get_text("blocks")
                
                for block in blocks:
                    text = block[4].strip()
                    
                    # Only add non-empty text blocks
                    if text:
                        data.append({
                            "page": page_no,
                            "text": text,
                            "bbox": {
                                "x0": block[0],
                                "y0": block[1],
                                "x1": block[2],
                                "y1": block[3]
                            }
                        })
        
        return data
    
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def extract_text_simple(pdf_path: str) -> str:
    """
    Extract all text from PDF as a single string
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Complete text content of the PDF
    """
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def get_pdf_page_count(pdf_path: str) -> int:
    """
    Get the total number of pages in a PDF
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Number of pages
    """
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
