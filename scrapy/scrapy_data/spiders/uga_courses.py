import re
from typing import List
import scrapy
from scrapy.http import Response

class UgaCoursesSpider(scrapy.Spider):
    name = 'uga_courses'
    allowed_domains = ['bulletin.uga.edu']
    
    custom_settings = {
        'FEEDS': {
            'output/uga_courses.json': {
                'format': 'json',
                'encoding': 'utf8',
                'indent': 2,
            },
        },
        'DOWNLOAD_DELAY': 3.0,  # Be extra polite to the server
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'ROBOTSTXT_OBEY': False,
    }
    
    # Realistic browser headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    start_urls: List[str] = ['https://bulletin.uga.edu/CoursesHome.aspx']
    start_url = 'https://bulletin.uga.edu/CoursesHome.aspx'

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                headers=self.headers,
                callback=self.parse
            )

    def parse(self, response: Response):
        self.logger.info(f"Starting course scraping from: {response.url}")
        
        # Extract all subject prefixes
        subjects = response.css('select#ddlAllPrefixes > option ::attr(value)').extract()
        self.logger.info(f"Found {len(subjects)} subject prefixes")
        
        # Only process the 'CSCI' prefix for testing
        for subject in subjects:
            self.logger.info(f"Processing subject: {subject}")
            viewstate = response.css('input#__VIEWSTATE::attr(value)').extract_first() or ''
            eventvalidation = response.css('input#__EVENTVALIDATION::attr(value)').extract_first() or ''
            yield scrapy.FormRequest(
                self.start_url,
                formdata={
                    'ddlAllPrefixes':    subject,
                    '__VIEWSTATE':       viewstate,
                    '__EVENTVALIDATION': eventvalidation
                },
                headers=self.headers,
                callback=self.parse_subjects,
                meta={'subject': subject}
            )

    def parse_subjects(self, response: Response):
        subject = response.meta.get('subject', 'Unknown')
        self.logger.info(f"Processing courses for subject: {subject}")
        
        courses = response.css('select#ddlAllCourses > option ::attr(value)').extract()
        
        for course in courses:
            # Skip over "Select a Course" option
            if course == '-1':
                continue
            # Skip over options that aren't "All Courses". Could remove this to get more detailed information.
            if course != '0':
                continue
                
            # Get form values, ensuring they're not None
            viewstate = response.css('input#__VIEWSTATE::attr(value)').extract_first() or ''
            eventvalidation = response.css('input#__EVENTVALIDATION::attr(value)').extract_first() or ''
            
            yield scrapy.FormRequest(
                self.start_url,
                formdata={
                    'ddlAllCourses':     course,
                    '__VIEWSTATE':       viewstate,
                    '__EVENTVALIDATION': eventvalidation
                },
                headers=self.headers,
                callback=self.parse_result,
                meta={'subject': subject}
            )

    # Regex patterns for extracting course information
    course_id = re.compile(r'Course ID:\n(.*?)\n')
    credit_hours = re.compile(r'Course ID:\n(?:.*?)\n\. (.*?)\n')
    course_title = re.compile(r'Course Title:\n(.*?)\n')
    course_description = re.compile(r'Course\nDescription:\n(.*?)\n')
    athena_title = re.compile(r'Athena Title:\n(.*?)\n')
    duplicate_credit = re.compile(r'Duplicate Credit:\n(.*?)\n')
    period = re.compile(r'Semester Course\nOffered:\n(.*?)\n')
    nontraditional = re.compile(r'Nontraditional Format:\n(.*?)\n')
    grading_system = re.compile(r'Grading System:\n(.*?)\n')

    def parse_result(self, response: Response):
        subject = response.meta.get('subject', 'Unknown')
        self.logger.info(f"Extracting course details for subject: {subject}")
        
        def try_search(pattern, string):
            try:
                result = re.search(pattern, string)
                return result.group(1).strip() if result else None
            except AttributeError:
                return None

        course_count = 0
        for course_table in response.css("table.courseresultstable"):
            course_info: List = course_table.css("td.courseinfo ::text").extract()

            if len(course_info) == 0:
                continue

            joined = '\n'.join(course_info) + '\n'
            joined = joined.replace('\r', ' ')

            result = {
                'subject': subject,
                'course_id':          try_search(self.course_id, joined),
                'credit_hours':       try_search(self.credit_hours, joined),
                'course_title':       try_search(self.course_title, joined),
                'course_description': try_search(self.course_description, joined),
                'athena_title':       try_search(self.athena_title, joined),
                'duplicate_credit':   try_search(self.duplicate_credit, joined),
                'period':             try_search(self.period, joined),
                'nontraditional':     try_search(self.nontraditional, joined),
                'grading_system':     try_search(self.grading_system, joined),
            }
            
            # Only yield if we have at least a course_id
            if result['course_id']:
                course_count += 1
                yield result
        
        self.logger.info(f"Extracted {course_count} courses for subject: {subject}") 