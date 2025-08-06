import scrapy
import re
from urllib.parse import urljoin

class UgaUniversityInfoSpider(scrapy.Spider):
    name = 'uga_university_info'
    allowed_domains = ['bulletin.uga.edu']
    
    custom_settings = {
        'FEEDS': {
            'output/uga_university_info.json': {
                'format': 'json',
                'encoding': 'utf8',
                'indent': 2,
            },
        },
        'DOWNLOAD_DELAY': 1.0,
        'ROBOTSTXT_OBEY': False,
    }
    
    start_urls = ['https://bulletin.uga.edu/UniversityInfoHome']
    
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
        self.logger.info(f"Parsing university info page: {response.url}")
        
        # Extract general university information
        university_info = self.extract_university_info(response)
        
        # Yield the main university information
        yield university_info
        
        # Follow links to detailed information pages
        info_links = self.extract_info_links(response)
        
        for link in info_links:
            if self.is_valid_info_link(link):
                full_url = urljoin(response.url, link)
                yield scrapy.Request(
                    url=full_url,
                    headers=self.headers,
                    callback=self.parse_info_detail,
                    meta={'dont_cache': True}
                )

    def extract_university_info(self, response):
        """Extract general university information from the main page"""
        # Get page title
        title = response.css('title::text').get()
        
        # Get main content
        content = response.css('body ::text').getall()
        content = ' '.join([text.strip() for text in content if text.strip()])
        
        # Extract specific information sections
        sections = self.extract_info_sections(content)
        
        # Extract contact information
        contact_info = self.extract_contact_info(content)
        
        # Extract academic calendar information
        calendar_info = self.extract_calendar_info(content)
        
        return {
            'type': 'university_info_main',
            'url': response.url,
            'title': title,
            'content_preview': content[:2000] if content else None,
            'sections': sections,
            'contact_info': contact_info,
            'calendar_info': calendar_info
        }

    def extract_info_sections(self, content):
        """Extract different information sections from content"""
        sections = {}
        
        # Look for specific sections
        section_patterns = {
            'academic_regulations': r'Academic\s+Regulations?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            'admissions': r'Admissions?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            'general_education': r'General\s+Education:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            'graduation_requirements': r'Graduation\s+Requirements?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            'student_services': r'Student\s+Services?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            'financial_aid': r'Financial\s+Aid:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
        
        return sections

    def extract_contact_info(self, content):
        """Extract contact information from content"""
        contact_info = {}
        
        # Look for email addresses
        email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        emails = re.findall(email_pattern, content)
        if emails:
            contact_info['emails'] = emails
        
        # Look for phone numbers
        phone_pattern = r'(\d{3}-\d{3}-\d{4})'
        phones = re.findall(phone_pattern, content)
        if phones:
            contact_info['phone_numbers'] = phones
        
        # Look for office information
        office_pattern = r'Office\s+of\s+([^:]+):\s*([^.\n]+)'
        offices = re.findall(office_pattern, content, re.IGNORECASE)
        if offices:
            contact_info['offices'] = [{'name': office[0].strip(), 'info': office[1].strip()} for office in offices]
        
        return contact_info

    def extract_calendar_info(self, content):
        """Extract academic calendar information"""
        calendar_info = {}
        
        # Look for calendar-related information
        calendar_patterns = {
            'semester_dates': r'Semester\s+(?:Dates?|Schedule):\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            'exam_schedule': r'Exam\s+Schedule:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            'deadlines': r'Deadlines?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            'holidays': r'Holidays?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        }
        
        for calendar_type, pattern in calendar_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                calendar_info[calendar_type] = match.group(1).strip()
        
        return calendar_info

    def extract_info_links(self, response):
        """Extract links to detailed information pages"""
        # Look for links to university information pages
        info_links = []
        
        # Get all links
        all_links = response.css('a::attr(href)').getall()
        
        for link in all_links:
            if link and self.is_valid_info_link(link):
                info_links.append(link)
        
        return info_links

    def is_valid_info_link(self, link):
        """Check if a link is valid for university information"""
        if not link:
            return False
        
        # Skip JavaScript links
        if link.startswith('javascript:'):
            return False
        
        # Skip external links
        if link.startswith('http') and 'bulletin.uga.edu' not in link:
            return False
        
        # Look for university info related keywords
        info_keywords = [
            'university', 'University', 'academic', 'Academic', 'policy', 'Policy',
            'regulation', 'Regulation', 'admission', 'Admission', 'calendar', 'Calendar',
            'requirement', 'Requirement', 'service', 'Service', 'financial', 'Financial'
        ]
        
        return any(keyword in link for keyword in info_keywords)

    def parse_info_detail(self, response):
        """Parse individual university information detail pages"""
        self.logger.info(f"Parsing university info detail page: {response.url}")
        
        # Extract detailed information
        detail_info = self.extract_detail_info(response)
        
        if detail_info:
            yield detail_info

    def extract_detail_info(self, response):
        """Extract detailed information from a specific university info page"""
        # Get page title
        title = response.css('title::text').get()
        
        # Get main content
        content = response.css('body ::text').getall()
        content = ' '.join([text.strip() for text in content if text.strip()])
        
        # Determine the type of information based on URL or title
        info_type = self.determine_info_type(response.url, title)
        
        # Extract specific information based on type
        specific_info = self.extract_specific_info(content, info_type)
        
        return {
            'type': 'university_info_detail',
            'info_type': info_type,
            'url': response.url,
            'title': title,
            'content_preview': content[:1500] if content else None,
            'specific_info': specific_info
        }

    def determine_info_type(self, url, title):
        """Determine the type of university information based on URL or title"""
        url_lower = url.lower()
        title_lower = title.lower() if title else ''
        
        if 'admission' in url_lower or 'admission' in title_lower:
            return 'admissions'
        elif 'academic' in url_lower or 'academic' in title_lower:
            return 'academic_regulations'
        elif 'calendar' in url_lower or 'calendar' in title_lower:
            return 'academic_calendar'
        elif 'financial' in url_lower or 'financial' in title_lower:
            return 'financial_information'
        elif 'student' in url_lower or 'student' in title_lower:
            return 'student_services'
        elif 'general' in url_lower and 'education' in url_lower:
            return 'general_education'
        else:
            return 'general_information'

    def extract_specific_info(self, content, info_type):
        """Extract specific information based on the type"""
        specific_info = {}
        
        if info_type == 'admissions':
            specific_info = self.extract_admissions_info(content)
        elif info_type == 'academic_regulations':
            specific_info = self.extract_academic_regulations_info(content)
        elif info_type == 'academic_calendar':
            specific_info = self.extract_calendar_info(content)
        elif info_type == 'financial_information':
            specific_info = self.extract_financial_info(content)
        elif info_type == 'student_services':
            specific_info = self.extract_student_services_info(content)
        elif info_type == 'general_education':
            specific_info = self.extract_general_education_info(content)
        else:
            specific_info = self.extract_general_info(content)
        
        return specific_info

    def extract_admissions_info(self, content):
        """Extract admissions-specific information"""
        admissions_info = {}
        
        # Look for admission requirements
        req_patterns = [
            r'Admission\s+Requirements?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Requirements?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Application\s+Process:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        ]
        
        for pattern in req_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                admissions_info['requirements'] = match.group(1).strip()
                break
        
        return admissions_info

    def extract_academic_regulations_info(self, content):
        """Extract academic regulations information"""
        regulations_info = {}
        
        # Look for academic policies
        policy_patterns = [
            r'Academic\s+Policies?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Regulations?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Standards?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        ]
        
        for pattern in policy_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                regulations_info['policies'] = match.group(1).strip()
                break
        
        return regulations_info

    def extract_financial_info(self, content):
        """Extract financial information"""
        financial_info = {}
        
        # Look for financial aid information
        aid_patterns = [
            r'Financial\s+Aid:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Scholarships?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Tuition:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        ]
        
        for pattern in aid_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                financial_info['aid_info'] = match.group(1).strip()
                break
        
        return financial_info

    def extract_student_services_info(self, content):
        """Extract student services information"""
        services_info = {}
        
        # Look for student services
        service_patterns = [
            r'Student\s+Services?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Services?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Support:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        ]
        
        for pattern in service_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                services_info['services'] = match.group(1).strip()
                break
        
        return services_info

    def extract_general_education_info(self, content):
        """Extract general education information"""
        gen_ed_info = {}
        
        # Look for general education requirements
        gen_ed_patterns = [
            r'General\s+Education:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Core\s+Curriculum:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Requirements?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        ]
        
        for pattern in gen_ed_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                gen_ed_info['requirements'] = match.group(1).strip()
                break
        
        return gen_ed_info

    def extract_general_info(self, content):
        """Extract general information"""
        general_info = {}
        
        # Look for any structured information
        info_patterns = [
            r'Information:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Details?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Overview:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        ]
        
        for pattern in info_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                general_info['details'] = match.group(1).strip()
                break
        
        return general_info 