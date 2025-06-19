import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Text processing libraries
from bs4 import BeautifulSoup
import pandas as pd

# Content extraction libraries (for HTML)
try:
    from boilerpy3 import extractors
    BOILERPY3_AVAILABLE = True
except ImportError:
    BOILERPY3_AVAILABLE = False
    logging.warning("boilerpy3 not available. HTML processing will use fallback methods.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustTextCleaner:
    def __init__(self):
        # Initialize content extractors for HTML
        if BOILERPY3_AVAILABLE:
            self.article_extractor = extractors.ArticleExtractor()
            self.default_extractor = extractors.DefaultExtractor()
        
        # Common UGA-specific noise patterns
        self.uga_noise_patterns = [
            # Navigation and accessibility
            r'skip to main content',
            r'skip to main menu', 
            r'skip to spotlight region',
            r'skip to secondary region',
            r'skip to uga region',
            r'skip to tertiary region',
            r'skip to quaternary region',
            r'skip to unit footer',
            
            # Social media and external links
            r'facebook|twitter|instagram|snapchat|youtube|linkedin',
            r"school's (twitter|youtube|linkedin) (feed|channel|page)",
            
            # UGA-specific boilerplate
            r'© university of georgia',
            r'human trafficking notice',
            r'reporting hotline',
            r'privacy policy',
            r'login for faculty',
            r'give now',
            r'search this site',
            r'submit search',
            r'close',
            
            # Contact and address patterns
            r'school of computing 415 boyd research and education center university of georgia athens, ga 30602-7404',
            r'click here to learn more about giving',
            r'your gift is important to us and helps support critical',
            r'support us we appreciate your financial support',
            r'news newsletter events media contact',
            
            # Common navigation elements
            r'main menu',
            r'mini menu',
            r'search',
            r'menu',
            r'close',
            r'previous',
            r'next',
            r'>>',
            r'<<',
            
            # JavaScript and technical artifacts
            r'{\s*"path":.*?}',
            r'pluralDelimiter',
            r'suppressDeprecationErrors',
            r'google_analytics',
            r'flexslider',
            r'instances',
            r'optionsets',
        ]
        
        # Compile regex patterns for efficiency
        self.noise_regex = re.compile('|'.join(self.uga_noise_patterns), re.IGNORECASE)
        
    def detect_content_type(self, content: str) -> str:
        """
        Detect whether content is HTML or raw text.
        """
        if not content:
            return "empty"
            
        # Check for HTML indicators
        html_indicators = [
            r'<html',
            r'<!DOCTYPE',
            r'<head',
            r'<body',
            r'<div',
            r'<p>',
            r'<span',
            r'<a\s+href',
            r'<script',
            r'<style',
            r'<meta',
            r'<title>',
            r'<h[1-6]>',
            r'<ul>',
            r'<ol>',
            r'<li>',
            r'<table>',
            r'<tr>',
            r'<td>',
            r'<th>',
        ]
        
        html_pattern = re.compile('|'.join(html_indicators), re.IGNORECASE)
        if html_pattern.search(content):
            return "html"
        else:
            return "text"
    
    def extract_content_from_html(self, html_content: str, url: str) -> Optional[str]:
        """
        Extract main content from HTML using multiple strategies.
        """
        if not html_content or not html_content.strip():
            return None
            
        try:
            # Strategy 1: Try boilerpy3 article extractor (best for news/content pages)
            if BOILERPY3_AVAILABLE:
                extracted = self.article_extractor.get_content(html_content)
                if extracted and len(extracted.strip()) > 100:
                    return extracted.strip()
                    
                # Strategy 2: Try boilerpy3 default extractor
                extracted = self.default_extractor.get_content(html_content)
                if extracted and len(extracted.strip()) > 100:
                    return extracted.strip()
            
            # Strategy 3: Fallback to BeautifulSoup with smart selection
            return self._fallback_extraction(html_content)
            
        except Exception as e:
            logger.warning(f"Error extracting content from {url}: {e}")
            return self._fallback_extraction(html_content)
    
    def _fallback_extraction(self, html_content: str) -> Optional[str]:
        """
        Fallback extraction using BeautifulSoup with smart content selection.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Try to find main content areas
            main_selectors = [
                'main',
                '[role="main"]',
                '.main-content',
                '.content',
                '#content',
                '.post-content',
                '.entry-content'
            ]
            
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    text = main_content.get_text(separator=' ', strip=True)
                    if len(text) > 100:
                        return text
            
            # If no main content found, get body text
            body = soup.find('body')
            if body:
                return body.get_text(separator=' ', strip=True)
                
        except Exception as e:
            logger.warning(f"Fallback extraction failed: {e}")
            
        return None
    
    def clean_extracted_text(self, text: str, url: str) -> Optional[str]:
        """
        Clean extracted text by removing noise patterns and normalizing.
        """
        if not text or not text.strip():
            return None
            
        # Remove noise patterns
        text = self.noise_regex.sub('', text)
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short content (likely noise)
        if len(text) < 100:
            return None
            
        # Remove content that's mostly punctuation or numbers
        if len(re.sub(r'[^\w\s]', '', text)) < len(text) * 0.3:
            return None
            
        # Remove content that's mostly repeated words (likely navigation)
        words = text.split()
        if len(set(words)) < len(words) * 0.4:
            return None
            
        # Remove content that's mostly single characters or very short words
        if len([w for w in words if len(w) > 2]) < len(words) * 0.6:
            return None
            
        return text
    
    def process_content(self, content: str, url: str) -> Optional[str]:
        """
        Main processing function that detects content type and applies appropriate cleaning.
        """
        if not content:
            return None
            
        content_type = self.detect_content_type(content)
        logger.debug(f"Detected content type for {url}: {content_type}")
        
        if content_type == "html":
            # Extract content from HTML first, then clean
            extracted_text = self.extract_content_from_html(content, url)
            if extracted_text:
                return self.clean_extracted_text(extracted_text, url)
        elif content_type == "text":
            # Clean raw text directly
            return self.clean_extracted_text(content, url)
        else:
            logger.warning(f"Unknown content type for {url}")
            
        return None
    
    def extract_metadata(self, text: str, url: str, title: str = "") -> Dict:
        """
        Extract useful metadata from the page.
        """
        metadata = {
            'source': url,
            'title': title,
            'department': '',
            'content_type': 'general'
        }
        
        # Try to determine department from URL or content
        if 'cs.uga.edu' in url:
            metadata['department'] = 'Computer Science'
        elif 'english' in url or 'english.uga.edu' in url:
            metadata['department'] = 'English'
        elif 'bulletin' in url:
            metadata['department'] = 'Academic Bulletin'
        
        # Determine content type based on URL patterns and content
        if any(pattern in url.lower() for pattern in ['program', 'degree', 'major']):
            metadata['content_type'] = 'program_info'
        elif any(pattern in url.lower() for pattern in ['contact', 'directory', 'people']):
            metadata['content_type'] = 'contact_info'
        elif any(pattern in url.lower() for pattern in ['news', 'events', 'announcement']):
            metadata['content_type'] = 'news_events'
        elif any(pattern in url.lower() for pattern in ['course', 'class']):
            metadata['content_type'] = 'course_info'
        elif any(pattern in url.lower() for pattern in ['research', 'lab']):
            metadata['content_type'] = 'research_info'
            
        return metadata

def load_scraped_data(input_dir: str) -> List[Dict]:
    """
    Load all JSON files from the scrapy output directory.
    """
    data = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory {input_dir} not found")
    
    for json_file in input_path.glob("*.json"):
        logger.info(f"Loading {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
                else:
                    data.append(file_data)
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    return data

def main():
    """
    Main cleaning pipeline.
    """
    # Initialize cleaner
    cleaner = RobustTextCleaner()
    
    # Load scraped data
    logger.info("Loading scraped data...")
    scraped_data = load_scraped_data("./scrapy/output")
    logger.info(f"Loaded {len(scraped_data)} pages")
    
    # Process each page
    cleaned_data = []
    
    for i, page_data in enumerate(scraped_data):
        if i % 100 == 0:
            logger.info(f"Processing page {i+1}/{len(scraped_data)}")
        
        url = page_data.get('url', '')
        title = page_data.get('title', '')
        
        # Get content (could be HTML or raw text)
        content = page_data.get('body', '')
        
        if not content:
            continue
        
        # Process content (automatically detects HTML vs text)
        cleaned_text = cleaner.process_content(content, url)
        if not cleaned_text:
            continue
        
        # Extract metadata
        metadata = cleaner.extract_metadata(cleaned_text, url, title)
        
        # Store results
        cleaned_data.append({
            'source': url,
            'title': metadata['title'],
            'department': metadata['department'],
            'content_type': metadata['content_type'],
            'cleaned_text': cleaned_text,
            'text_length': len(cleaned_text)
        })
    
    # Save results
    logger.info(f"Cleaned {len(cleaned_data)} pages")
    
    # Save as CSV
    df = pd.DataFrame(cleaned_data)
    output_dir = Path("cleaned_data_csv")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "cleaned_data.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved cleaned data to {csv_path}")
    
    # Save as JSON for easier processing
    json_path = output_dir / "cleaned_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved cleaned data to {json_path}")
    
    # Print summary statistics
    print("\n=== Cleaning Summary ===")
    print(f"Total pages processed: {len(scraped_data)}")
    print(f"Successfully cleaned: {len(cleaned_data)}")
    print(f"Success rate: {len(cleaned_data)/len(scraped_data)*100:.1f}%")
    
    if cleaned_data:
        avg_length = sum(item['text_length'] for item in cleaned_data) / len(cleaned_data)
        print(f"Average text length: {avg_length:.0f} characters")
        
        # Department breakdown
        dept_counts = {}
        for item in cleaned_data:
            dept = item['department'] or 'Unknown'
            dept_counts[dept] = dept_counts.get(dept, 0) + 1
        
        print("\nDepartment breakdown:")
        for dept, count in sorted(dept_counts.items()):
            print(f"  {dept}: {count} pages")
            
        # Content type breakdown
        content_counts = {}
        for item in cleaned_data:
            content_type = item['content_type']
            content_counts[content_type] = content_counts.get(content_type, 0) + 1
        
        print("\nContent type breakdown:")
        for content_type, count in sorted(content_counts.items()):
            print(f"  {content_type}: {count} pages")

if __name__ == "__main__":
    main()