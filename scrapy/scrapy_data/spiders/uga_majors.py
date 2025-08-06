import scrapy
import re
from urllib.parse import urljoin

class UgaMajorsSpider(scrapy.Spider):
    name = 'uga_majors'
    allowed_domains = ['bulletin.uga.edu']
    
    custom_settings = {
        'FEEDS': {
            'output/uga_majors.json': {
                'format': 'json',
                'encoding': 'utf8',
                'indent': 2,
            },
        },
        'DOWNLOAD_DELAY': 1.0,
        'ROBOTSTXT_OBEY': False,
    }
    
    start_urls = ['https://bulletin.uga.edu/MajorsHome']
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                headers=self.headers,
                callback=self.parse,
                meta={'dont_cache': True}
            )

    def parse(self, response):
        self.logger.info(f"Parsing majors page: {response.url}")
        
        # Extract all major links and information
        majors = self.extract_majors_from_page(response)
        
        # Yield each major found
        for major in majors:
            yield major
        
        # Follow links to detailed major pages
        major_links = response.css('a[href*="Major"]::attr(href)').getall()
        for link in major_links:
            if link and self.is_valid_major_link(link):
                full_url = urljoin(response.url, link)
                yield scrapy.Request(
                    url=full_url,
                    headers=self.headers,
                    callback=self.parse_major_detail,
                    meta={'dont_cache': True}
                )

    def extract_majors_from_page(self, response):
        """Extract major information from the current page"""
        majors = []
        
        # Get all text content
        page_text = response.css('body ::text').getall()
        page_text = ' '.join([text.strip() for text in page_text if text.strip()])
        
        # Look for major patterns in the text
        major_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*-\s*([A-Z]+\.[A-Z]+)',  # "Computer Science - B.S."
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*-\s*([A-Z]+\.[A-Z]+\.[A-Z]+)',  # "Computer Science - B.S.C.S."
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*-\s*([A-Z]+\.[A-Z]+\.[A-Z]+\.[A-Z]+)',  # "Computer Science - B.S.C.S.E."
        ]
        
        for pattern in major_patterns:
            matches = re.findall(pattern, page_text)
            for match in matches:
                major_name, degree_type = match
                major_name = major_name.strip()
                degree_type = degree_type.strip()
                
                # Filter out generic terms
                if len(major_name) > 3 and not self.is_generic_term(major_name):
                    majors.append({
                        'major_name': major_name,
                        'degree_type': degree_type,
                        'source': 'page_text',
                        'url': response.url
                    })
        
        # Also look for major links
        major_links = response.css('a::text').getall()
        for link_text in major_links:
            if link_text and ' - ' in link_text:
                parts = link_text.split(' - ')
                if len(parts) == 2:
                    major_name = parts[0].strip()
                    degree_type = parts[1].strip()
                    
                    if len(major_name) > 3 and not self.is_generic_term(major_name):
                        majors.append({
                            'major_name': major_name,
                            'degree_type': degree_type,
                            'source': 'link_text',
                            'url': response.url
                        })
        
        return majors

    def is_generic_term(self, term):
        """Check if a term is too generic to be a major name"""
        generic_terms = {
            'Home', 'Major', 'Program', 'Degree', 'Bachelor', 'Master', 'Doctor',
            'Undergraduate', 'Graduate', 'Professional', 'Select', 'View', 'Compare',
            'Campuses', 'Resources', 'Athena', 'Explore'
        }
        return term in generic_terms

    def is_valid_major_link(self, link):
        """Check if a link is valid for following"""
        if not link:
            return False
        
        # Skip JavaScript links
        if link.startswith('javascript:'):
            return False
        
        # Skip external links
        if link.startswith('http') and 'bulletin.uga.edu' not in link:
            return False
        
        # Look for major-related keywords
        major_keywords = ['major', 'Major', 'program', 'Program', 'degree', 'Degree']
        return any(keyword in link for keyword in major_keywords)

    def parse_major_detail(self, response):
        """Parse individual major detail pages"""
        self.logger.info(f"Parsing major detail page: {response.url}")
        
        # Extract major information from detail page
        major_info = self.extract_major_detail_info(response)
        
        if major_info:
            yield major_info

    def extract_major_detail_info(self, response):
        """Extract detailed major information from a specific major page"""
        # Get page title
        title = response.css('title::text').get()
        
        # Get main content
        content = response.css('body ::text').getall()
        content = ' '.join([text.strip() for text in content if text.strip()])
        
        # Look for major name and degree type in title or content
        major_name = None
        degree_type = None
        
        # Try to extract from title first
        if title:
            title_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*-\s*([A-Z]+\.[A-Z]+(?:\.[A-Z]+)*)', title)
            if title_match:
                major_name = title_match.group(1).strip()
                degree_type = title_match.group(2).strip()
        
        # If not found in title, try content
        if not major_name:
            content_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*-\s*([A-Z]+\.[A-Z]+(?:\.[A-Z]+)*)', content)
            if content_match:
                major_name = content_match.group(1).strip()
                degree_type = content_match.group(2).strip()
        
        # Extract requirements and other details
        requirements = self.extract_requirements(content)
        description = self.extract_description(content)
        
        return {
            'type': 'major_detail',
            'url': response.url,
            'title': title,
            'major_name': major_name,
            'degree_type': degree_type,
            'description': description,
            'requirements': requirements,
            'content_preview': content[:1000] if content else None
        }

    def extract_requirements(self, content):
        """Extract major requirements from content"""
        requirements = []
        
        # Look for requirement patterns
        req_patterns = [
            r'Requirements?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Required\s+(?:courses?|hours?):\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Total\s+(?:hours?|credits?):\s*(\d+)',
        ]
        
        for pattern in req_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if match.strip():
                    requirements.append(match.strip())
        
        return requirements

    def extract_description(self, content):
        """Extract major description from content"""
        # Look for description patterns
        desc_patterns = [
            r'Description:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Overview:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'About\s+this\s+major:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        ]
        
        for pattern in desc_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None 