import scrapy
from urllib.parse import urlparse

class CSSpider(scrapy.Spider):
    name = 'uga_cs'
    start_urls = ['https://www.cs.uga.edu/']

    def parse(self, response):
        # Extract all the text content from the page
        yield {
            'url': response.url,
            'title': response.xpath('//title/text()').get(),
            'body': ' '.join(response.css('body ::text').getall()).strip(),
        }

        for next_page in response.css('a::attr(href)').getall():
            next_page = response.urljoin(next_page)

            # Check if the link is within the cs.uga.edu domain
            if self.is_valid_domain(next_page):
                yield scrapy.Request(next_page, callback=self.parse)

    def is_valid_domain(self, url):
        # Parse the URL and extract the domain
        parsed_url = urlparse(url)
        
        return parsed_url.netloc == 'www.cs.uga.edu'
