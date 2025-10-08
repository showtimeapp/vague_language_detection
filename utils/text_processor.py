"""
Text Processing Module
Handles paragraph and sentence segmentation
"""

import nltk
from typing import List, Dict
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def segment_paragraphs(text: str, min_length: int = 50) -> List[str]:
    """
    Segment text into paragraphs
    
    Args:
        text: Input text to segment
        min_length: Minimum character length for a paragraph
        
    Returns:
        List of paragraphs
    """
    # Split by double newlines (common paragraph separator)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Clean up paragraphs
    cleaned_paragraphs = []
    buffer = ""
    
    for para in paragraphs:
        # Remove extra whitespace
        para = re.sub(r'\s+', ' ', para).strip()
        
        # Skip empty paragraphs
        if len(para) < 10:
            continue
        
        # If paragraph is too short, combine with buffer
        if len(para) < min_length and buffer:
            buffer += " " + para
        elif len(para) < min_length:
            buffer = para
        else:
            # Add buffer if exists
            if buffer:
                cleaned_paragraphs.append(buffer)
                buffer = ""
            cleaned_paragraphs.append(para)
    
    # Add remaining buffer
    if buffer:
        cleaned_paragraphs.append(buffer)
    
    # If no paragraphs found (single block of text), split by sentences
    if len(cleaned_paragraphs) == 0 or (len(cleaned_paragraphs) == 1 and len(cleaned_paragraphs[0]) > 1000):
        return segment_by_length(text, target_length=500)
    
    return cleaned_paragraphs


def segment_by_length(text: str, target_length: int = 500) -> List[str]:
    """
    Segment text into chunks of approximately target_length
    Tries to break at sentence boundaries
    
    Args:
        text: Input text to segment
        target_length: Target length for each chunk
        
    Returns:
        List of text chunks
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence exceeds target, save current chunk
        if len(current_chunk) + len(sentence) > target_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def segment_sentences(text: str) -> List[str]:
    """
    Segment text into individual sentences
    
    Args:
        text: Input text to segment
        
    Returns:
        List of sentences
    """
    # Use NLTK's sentence tokenizer
    sentences = nltk.sent_tokenize(text)
    
    # Clean up sentences
    cleaned_sentences = []
    for sent in sentences:
        # Remove extra whitespace
        sent = re.sub(r'\s+', ' ', sent).strip()
        # Only include sentences with meaningful content
        if len(sent) > 10:  # Minimum sentence length
            cleaned_sentences.append(sent)
    
    return cleaned_sentences


def create_context_window(sentences: List[str], index: int, window_size: int = 2) -> Dict[str, str]:
    """
    Create a context window around a specific sentence
    
    Args:
        sentences: List of all sentences
        index: Index of the target sentence
        window_size: Number of sentences before/after to include
        
    Returns:
        Dictionary with previous, current, and next context
    """
    total = len(sentences)
    
    # Get previous context
    prev_start = max(0, index - window_size)
    prev_context = " ".join(sentences[prev_start:index]) if index > 0 else ""
    
    # Current sentence
    current = sentences[index]
    
    # Get next context
    next_end = min(total, index + window_size + 1)
    next_context = " ".join(sentences[index + 1:next_end]) if index < total - 1 else ""
    
    return {
        "previous": prev_context,
        "current": current,
        "next": next_context
    }


def map_paragraphs_to_blocks(paragraphs: List[str], blocks: List[Dict]) -> List[Dict]:
    """
    Map paragraphs back to their PDF blocks for location tracking
    
    Args:
        paragraphs: List of paragraphs
        blocks: List of text blocks with location data from PDF
        
    Returns:
        List of paragraphs with their page numbers and bounding boxes
    """
    paragraph_data = []
    
    for paragraph in paragraphs:
        # Find which block this paragraph starts in
        found = False
        
        # Try to find the paragraph in blocks
        para_start = paragraph[:50]  # First 50 chars for matching
        
        for block in blocks:
            if para_start in block["text"] or block["text"] in paragraph:
                paragraph_data.append({
                    "paragraph": paragraph,
                    "page": block["page"],
                    "bbox": block["bbox"]
                })
                found = True
                break
        
        # If not found in any block, assign to first block (fallback)
        if not found and blocks:
            paragraph_data.append({
                "paragraph": paragraph,
                "page": blocks[0]["page"],
                "bbox": blocks[0]["bbox"]
            })
    
    return paragraph_data


def map_sentences_to_blocks(sentences: List[str], blocks: List[Dict]) -> List[Dict]:
    """
    Map individual sentences back to their PDF blocks for location tracking
    
    Args:
        sentences: List of sentences
        blocks: List of text blocks with location data from PDF
        
    Returns:
        List of sentences with their page numbers and bounding boxes
    """
    sentence_data = []
    
    # Combine all block texts
    all_text = " ".join([block["text"] for block in blocks])
    
    for sentence in sentences:
        # Find which block this sentence belongs to
        found = False
        for block in blocks:
            if sentence in block["text"]:
                sentence_data.append({
                    "sentence": sentence,
                    "page": block["page"],
                    "bbox": block["bbox"]
                })
                found = True
                break
        
        # If not found in any block, assign to first block (fallback)
        if not found and blocks:
            sentence_data.append({
                "sentence": sentence,
                "page": blocks[0]["page"],
                "bbox": blocks[0]["bbox"]
            })
    
    return sentence_data
