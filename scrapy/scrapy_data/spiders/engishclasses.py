import scrapy

class CourseSpider(scrapy.Spider):
    name = "course_spider"

    # Starting URL for the website
    start_urls = [
        'https://www.terry.uga.edu/'  # Base URL of the site
    ]

    def parse(self, response):
        # Extract main content (course descriptions, prerequisites, etc.)
        main_content = content.xpath('//div[@id="main"]//article')
        
        # Iterate over main content and extract details
        for content in main_content:
            title = content.xpath('.//h1//text()').get()
            description = response.xpath('/html/body/div[1]/main/article/div').getall()
            prerequisites = content.xpath('.//h4[contains(text(), "Prerequisites")]/following-sibling::p//text()').getall()
            course_texts = content.xpath('.//h4[contains(text(), "Course Texts")]/following-sibling::p//text()').getall()

            # Yield the scraped data
            yield {
                'title': title.strip() if title else 'N/A',
                'description': ' '.join(description).strip() if description else 'N/A',
                'prerequisites': ' '.join(prerequisites).strip() if prerequisites else 'N/A',
                'course_texts': ' '.join(course_texts).strip() if course_texts else 'N/A',
            }

        for next_page in response.css('a::attr(href)').getall():
            if next_page and next_page.startswith('/') and not next_page.startswith('#'):
                yield response.follow(next_page, self.parse)

