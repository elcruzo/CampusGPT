"""
Text processing utilities for CampusGPT
Query preprocessing and response postprocessing
"""

import re
import string
from typing import List, Optional
import unicodedata


def preprocess_query(text: str) -> str:
    """
    Preprocess user query for better model performance
    
    Args:
        text: Input query text
        
    Returns:
        Preprocessed query text
    """
    if not text:
        return ""
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Fix common abbreviations
    text = _expand_abbreviations(text)
    
    # Remove excessive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Ensure proper sentence ending
    if text and text[-1] not in '.!?':
        text += '?'
    
    return text


def postprocess_response(text: str) -> str:
    """
    Postprocess model response for better quality
    
    Args:
        text: Model generated response
        
    Returns:
        Postprocessed response text
    """
    if not text:
        return ""
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove incomplete sentences at the end
    text = _remove_incomplete_sentences(text)
    
    # Fix spacing around punctuation
    text = _fix_punctuation_spacing(text)
    
    # Remove repetitive phrases
    text = _remove_repetition(text)
    
    # Ensure proper capitalization
    text = _fix_capitalization(text)
    
    return text


def _expand_abbreviations(text: str) -> str:
    """Expand common abbreviations"""
    abbreviations = {
        r'\bfaq\b': 'FAQ',
        r'\bcs\b': 'Computer Science',
        r'\bit\b': 'IT',
        r'\bgpa\b': 'GPA',
        r'\bfafsa\b': 'FAFSA',
        r'\badm\b': 'admissions',
        r'\breg\b': 'registration',
        r'\bfinaid\b': 'financial aid',
        r'\bdorm\b': 'dormitory',
        r'\bres hall\b': 'residence hall',
        r'\bcampus\b': 'campus'
    }
    
    for abbrev, expansion in abbreviations.items():
        text = re.sub(abbrev, expansion, text, flags=re.IGNORECASE)
    
    return text


def _remove_incomplete_sentences(text: str) -> str:
    """Remove incomplete sentences at the end"""
    if not text:
        return text
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    if len(sentences) <= 1:
        return text
    
    # Check if last sentence is incomplete (very short or doesn't end properly)
    last_sentence = sentences[-1].strip()
    
    # Remove if it's too short or looks incomplete
    if len(last_sentence) < 10 or not last_sentence:
        sentences = sentences[:-1]
        # Reconstruct text
        text = '. '.join(s.strip() for s in sentences if s.strip())
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
    
    return text


def _fix_punctuation_spacing(text: str) -> str:
    """Fix spacing around punctuation"""
    # Add space after punctuation if missing
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([,;:])([^\s])', r'\1 \2', text)
    
    # Remove extra spaces before punctuation
    text = re.sub(r'\s+([.!?,:;])', r'\1', text)
    
    # Fix parentheses spacing
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    
    return text


def _remove_repetition(text: str, max_repeat: int = 3) -> str:
    """Remove repetitive phrases"""
    # Remove repeated words (more than max_repeat times)
    words = text.split()
    filtered_words = []
    word_count = {}
    
    for word in words:
        word_lower = word.lower().strip(string.punctuation)
        word_count[word_lower] = word_count.get(word_lower, 0) + 1
        
        if word_count[word_lower] <= max_repeat:
            filtered_words.append(word)
    
    text = ' '.join(filtered_words)
    
    # Remove repeated phrases (3+ words)
    # This is a simple approach - could be more sophisticated
    text = re.sub(r'\b(\w+\s+\w+\s+\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
    
    return text


def _fix_capitalization(text: str) -> str:
    """Fix capitalization issues"""
    if not text:
        return text
    
    # Capitalize first letter
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    # Capitalize after sentence endings
    text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    
    # Fix common proper nouns
    proper_nouns = [
        'university', 'college', 'campus', 'library', 'student union',
        'dining hall', 'residence hall', 'admissions office',
        'financial aid', 'registrar', 'bursar', 'fafsa'
    ]
    
    for noun in proper_nouns:
        text = re.sub(rf'\b{noun}\b', noun.title(), text, flags=re.IGNORECASE)
    
    return text


def extract_keywords(text: str) -> List[str]:
    """
    Extract keywords from text for categorization
    
    Args:
        text: Input text
        
    Returns:
        List of extracted keywords
    """
    # Simple keyword extraction
    text = text.lower()
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those',
        'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his',
        'she', 'her', 'hers', 'it', 'its', 'we', 'us', 'our', 'ours', 'they',
        'them', 'their', 'theirs'
    }
    
    # Extract words
    words = re.findall(r'\b\w+\b', text)
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    return keywords


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple similarity between two texts
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Extract keywords from both texts
    keywords1 = set(extract_keywords(text1))
    keywords2 = set(extract_keywords(text2))
    
    if not keywords1 and not keywords2:
        return 1.0 if text1.strip() == text2.strip() else 0.0
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = keywords1.intersection(keywords2)
    union = keywords1.union(keywords2)
    
    return len(intersection) / len(union) if union else 0.0


def truncate_text(text: str, max_length: int, preserve_words: bool = True) -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Input text
        max_length: Maximum character length
        preserve_words: Whether to preserve word boundaries
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    if preserve_words:
        # Find last complete word within limit
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we don't lose too much
            return truncated[:last_space] + '...'
    
    return text[:max_length - 3] + '...'