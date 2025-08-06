import json
import os
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import re

try:
    from boilerpy3 import extractors
    BOILERPY3_AVAILABLE = True
except ImportError:
    BOILERPY3_AVAILABLE = False

def detect_content_type(content: str) -> str:
    if not content:
        return "empty"
    html_indicators = [
        r'<html', r'<!DOCTYPE', r'<head', r'<body', r'<div', r'<p>', r'<span', r'<a\s+href',
        r'<script', r'<style', r'<meta', r'<title>', r'<h[1-6]>', r'<ul>', r'<ol>', r'<li>',
        r'<table>', r'<tr>', r'<td>', r'<th>'
    ]
    html_pattern = re.compile('|'.join(html_indicators), re.IGNORECASE)
    if html_pattern.search(content):
        return "html"
    else:
        return "text"

def extract_content_from_html(html_content: str) -> str:
    if not html_content or not html_content.strip():
        return ""
    try:
        if BOILERPY3_AVAILABLE:
            article_extractor = extractors.ArticleExtractor()
            extracted = article_extractor.get_content(html_content)
            if extracted and len(extracted.strip()) > 100:
                return extracted.strip()
            default_extractor = extractors.DefaultExtractor()
            extracted = default_extractor.get_content(html_content)
            if extracted and len(extracted.strip()) > 100:
                return extracted.strip()
        # Fallback to BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        main_selectors = [
            'main', '[role="main"]', '.main-content', '.content', '#content', '.post-content', '.entry-content'
        ]
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
                if len(text) > 100:
                    return text
        body = soup.find('body')
        if body:
            return body.get_text(separator=' ', strip=True)
    except Exception:
        pass
    return ""

def clean_extracted_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    # Remove UGA-specific and common boilerplate
    noise_patterns = [
        r'skip to main content', r'skip to main menu', r'skip to spotlight region',
        r'skip to secondary region', r'skip to uga region', r'skip to tertiary region',
        r'skip to quaternary region', r'skip to unit footer', r'facebook|twitter|instagram|snapchat|youtube|linkedin',
        r"school's (twitter|youtube|linkedin) (feed|channel|page)", r'© university of georgia',
        r'human trafficking notice', r'reporting hotline', r'privacy policy', r'login for faculty',
        r'give now', r'search this site', r'submit search', r'close', r'main menu', r'mini menu',
        r'search', r'menu', r'close', r'previous', r'next', r'>>', r'<<', r'{\s*"path":.*?}',
        r'pluralDelimiter', r'suppressDeprecationErrors', r'google_analytics', r'flexslider',
        r'instances', r'optionsets',
    ]
    noise_regex = re.compile('|'.join(noise_patterns), re.IGNORECASE)
    text = noise_regex.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_clean(input_dir: str) -> List[Dict[str, Any]]:
    data = []
    input_path = Path(input_dir)
    for json_file in input_path.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                file_data = json.load(f)
            except Exception:
                continue
            if isinstance(file_data, list):
                items = file_data
            else:
                items = [file_data]
            for item in items:
                # Handle already structured data (uga_courses, uga_rmp, etc.)
                if 'course_description' in item:
                    cleaned_text = item.get('course_description', '')
                elif 'reviews' in item and isinstance(item['reviews'], list):
                    cleaned_text = ' '.join([r['text'] for r in item['reviews'] if 'text' in r])
                else:
                    content = item.get('description') or item.get('body') or ''
                    content_type = detect_content_type(content)
                    if content_type == 'html':
                        extracted = extract_content_from_html(content)
                        cleaned_text = clean_extracted_text(extracted)
                    else:
                        cleaned_text = clean_extracted_text(content)
                data.append({
                    'cleaned_text': cleaned_text,
                    'source': str(json_file),
                    'raw': item
                })
    return data

if __name__ == "__main__":
    import sys
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "./scrapy/output"
    cleaned = load_and_clean(input_dir)
    print(f"Loaded and cleaned {len(cleaned)} items.") 