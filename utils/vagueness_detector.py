"""
Vagueness Detection Module
Uses Google Gemini API to detect and explain vague language
"""

import google.generativeai as genai
import json
from typing import Dict, List


def analyze_paragraph(paragraph: str, api_key: str, model_name: str = "gemini-2.5-flash") -> Dict:
    """
    Analyze a paragraph for vagueness using Gemini
    
    Args:
        paragraph: The paragraph to analyze
        api_key: Google Gemini API key
        model_name: Gemini model to use
        
    Returns:
        Dictionary with analysis results
    """
    
    # Configure API key for this request
    genai.configure(api_key=api_key)
    
    # Create the prompt for paragraph analysis
    prompt = f"""You are an expert linguist evaluating clarity and precision in academic and technical writing.

Task:
Analyze the following paragraph and determine if it contains vague, unclear, or imprecise language.

A paragraph is considered vague if it contains:
- Subjective qualifiers without specific metrics (e.g., "very effective", "quite fast", "significantly better")
- Unclear pronouns or ambiguous references
- Imprecise quantifiers (e.g., "many", "several", "some", "a number of")
- Vague time references (e.g., "soon", "recently", "in the near future", "for a while")
- Hedging language without proper justification (e.g., "might be", "possibly", "could potentially")
- General statements without supporting evidence or data
- Comparative statements without baselines (e.g., "more efficient", "better performance")

Paragraph to analyze:
"{paragraph}"

Instructions:
1. Identify if this paragraph contains vague or imprecise language
2. If vague, explain SPECIFICALLY which phrases or claims are vague and WHY
3. Suggest concrete ways to make it more precise (e.g., add specific numbers, metrics, or references)

Respond ONLY with valid JSON in this exact format:
{{"is_vague": true or false, "reason": "detailed explanation of what is vague", "suggestion": "specific improvements with examples"}}

If the paragraph is clear and precise with concrete details, metrics, or specific information, mark is_vague as false."""

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        # Parse JSON
        result = json.loads(response_text)
        
        return {
            "is_vague": result.get("is_vague", False),
            "reason": result.get("reason", "No reason provided"),
            "suggestion": result.get("suggestion", "No suggestion provided")
        }
    
    except json.JSONDecodeError as e:
        return {
            "is_vague": False,
            "reason": f"Error parsing response: {str(e)}",
            "suggestion": paragraph
        }
    
    except Exception as e:
        return {
            "is_vague": False,
            "reason": f"Error analyzing paragraph: {str(e)}",
            "suggestion": paragraph
        }


def analyze_sentence(sentence: str, api_key: str, context: str = "", model_name: str = "gemini-2.5-flash") -> Dict:
    """
    Analyze a single sentence for vagueness using Gemini
    
    Args:
        sentence: The sentence to analyze
        api_key: Google Gemini API key
        context: Optional context around the sentence
        model_name: Gemini model to use
        
    Returns:
        Dictionary with analysis results
    """
    
    # Configure API key for this request
    genai.configure(api_key=api_key)
    
    # Create the prompt
    prompt = f"""You are an expert linguist evaluating clarity in academic and technical writing.

Task:
1. Identify if the sentence below is vague, unclear, or imprecise.
2. If vague, explain WHY it is vague in 1-2 clear lines.
3. Suggest a more precise alternative.

A sentence is considered vague if it contains:
- Subjective qualifiers without metrics (e.g., "very good", "quite fast", "somewhat better")
- Unclear pronouns or references
- Ambiguous quantifiers (e.g., "many", "several", "some")
- Imprecise time references (e.g., "soon", "recently", "for a while")
- Hedging language without justification (e.g., "might", "possibly", "could be")

Sentence: "{sentence}"
{f'Context: {context}' if context else ''}

Respond ONLY with valid JSON in this exact format:
{{"is_vague": true or false, "reason": "explanation here", "suggestion": "improved version here"}}"""

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        # Parse JSON
        result = json.loads(response_text)
        
        return {
            "is_vague": result.get("is_vague", False),
            "reason": result.get("reason", "No reason provided"),
            "suggestion": result.get("suggestion", "No suggestion provided")
        }
    
    except json.JSONDecodeError as e:
        return {
            "is_vague": False,
            "reason": f"Error parsing response: {str(e)}",
            "suggestion": sentence
        }
    
    except Exception as e:
        return {
            "is_vague": False,
            "reason": f"Error analyzing sentence: {str(e)}",
            "suggestion": sentence
        }


def analyze_batch(items: List[str], api_key: str, progress_callback=None, model_name: str = "gemini-2.5-flash", use_paragraphs: bool = True) -> List[Dict]:
    """
    Analyze multiple items (sentences or paragraphs) in batch
    
    Args:
        items: List of text items to analyze
        api_key: Google Gemini API key
        progress_callback: Optional callback function for progress updates
        model_name: Gemini model to use
        use_paragraphs: If True, use paragraph analysis; if False, use sentence analysis
        
    Returns:
        List of analysis results
    """
    results = []
    total = len(items)
    analyze_func = analyze_paragraph if use_paragraphs else analyze_sentence
    
    for i, item in enumerate(items):
        result = analyze_func(item, api_key=api_key, model_name=model_name)
        results.append(result)
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback(i + 1, total)
    
    return results
