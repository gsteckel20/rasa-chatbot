import scrapy

class UgaBulletinSpider(scrapy.Spider):
    name = 'ugabulletin'
    start_urls = ['https://www.bulletin.uga.edu/']

    def parse(self, response):
        # Extract course information from UGA Bulletin
        for course in response.css('.course-list-item'):
            yield {
                'course_title': course.css('.course-title::text').get(),
                'course_description': course.css('.course-description::text').get(),
                'prerequisites': course.css('.course-prerequisites::text').get(),
                'credits': course.css('.course-credits::text').get(),
            }

        # Follow links to next pages of courses
        next_page = response.css('a.next-page::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)
