import scrapy
import json
import re
from urllib.parse import urlencode

class UgaRmpSpider(scrapy.Spider):
    name = "uga_rmp"
    allowed_domains = ["ratemyprofessors.com"]
    custom_settings = {
        'FEEDS': {
            'output/uga_rmp.json': {
                'format': 'json',
                'encoding': 'utf8',
                'indent': 2,
            },
        },
        'DOWNLOAD_DELAY': 1.0,  # Be polite
    }

    graphql_url = "https://www.ratemyprofessors.com/graphql"
    graphql_query = '''query TeacherSearchPaginationQuery(\n  $count: Int!\n  $cursor: String\n  $query: TeacherSearchQuery!\n) {\n  search: newSearch {\n    ...TeacherSearchPagination_search_1jWD3d\n  }\n}\n\nfragment CardFeedback_teacher on Teacher {\n  wouldTakeAgainPercent\n  avgDifficulty\n}\n\nfragment CardName_teacher on Teacher {\n  firstName\n  lastName\n}\n\nfragment TeacherCard_teacher on Teacher {\n  id\n  legacyId\n  avgRating\n  ...CardFeedback_teacher\n  ...CardName_teacher\n}\n\nfragment TeacherSearchPagination_search_1jWD3d on newSearch {\n  teachers(query: $query, first: $count, after: $cursor) {\n    didFallback\n    edges {\n      cursor\n      node {\n        ...TeacherCard_teacher\n        id\n        __typename\n      }\n    }\n    pageInfo {\n      hasNextPage\n      endCursor\n    }\n    resultCount\n    filters {\n      field\n      options {\n        value\n        id\n      }\n    }\n  }\n}\n'''
    school_id = "U2Nob29sLTExMDE="  # UGA
    # Provided cookie header (redact if needed)
    cookie_header = "RMP_AUTH_COOKIE_VERSION=v02; _pubcid=2b1895b0-3204-4e0c-8a7d-bbcca1407763; pjs-unifiedid=%7B%22TDID%22%3A%22b60b6a9a-9f2d-4aac-a421-c679da2a37ea%22%2C%22TDID_LOOKUP%22%3A%22TRUE%22%2C%22TDID_CREATED_AT%22%3A%222025-05-23T16%3A07%3A10%22%7D; pjs-unifiedid_cst=zix7LPQsHA%3D%3D; _ga=GA1.1.1937586284.1750694830; _cc_id=3fbf609a3d22fc0535953afe580c2088; panoramaId_expiry=1751299631015; panoramaId=9dc4fb0fb360fbf0885356e0cef916d53938532a7e07b9b9e613e8bedbbc3fba; panoramaIdType=panoIndiv; _li_dcdm_c=.ratemyprofessors.com; _lc2_fpi=5ee24c8f6482--01jyerp942nps54sy0a9ssabbg; _lc2_fpi_meta=%7B%22w%22%3A1750694831234%7D; _lr_env_src_ats=false; pbjs-unifiedid=%7B%22TDID%22%3A%22b60b6a9a-9f2d-4aac-a421-c679da2a37ea%22%2C%22TDID_LOOKUP%22%3A%22TRUE%22%2C%22TDID_CREATED_AT%22%3A%222025-05-23T16%3A07%3A10%22%7D; pbjs-unifiedid_cst=VyxHLMwsHQ%3D%3D; _au_1d=AU1D-0100-001750694831-RRW71ADQ-KNI8; logglytrackingsession=fdd228ea-e93c-4927-8092-82c4372e8ec4; _lc2_fpi_js=5ee24c8f6482--01jyerp942nps54sy0a9ssabbg; _li_ss=CgA; cid=Pk3VlzZ0H--20250623; cnx_userId=1-5590acab94e546ce81b88cca617d0d65; _hjSessionUser_1667000=eyJpZCI6IjgxNjQzMTNjLTAwYmEtNTU2ZS05ZDZkLWRkMDViNjcwM2I5NSIsImNyZWF0ZWQiOjE3NTA2OTQ4MzA3NDAsImV4aXN0aW5nIjp0cnVlfQ==; userSchoolId=U2Nob29sLTExMDE=; userSchoolLegacyId=1101; userSchoolName=University%20of%20Georgia; _iiq_fdata=%7B%22pcid%22%3A%220dd42514-1a17-4322-aefb-e3536de316ef%22%2C%22pcidDate%22%3A1750694832184%2C%22dbsaved%22%3A%22false%22%2C%22isOptedOut%22%3Afalse%7D; _iiq_ab_map=%7B%2295%22%3A%22A%22%7D; krg_uid=%7B%22v%22%3A%7B%22clientId%22%3A%2246ab5dec-ca6a-46c9-b677-cf4f3d66be67%22%2C%22userId%22%3A%22e75be359-faff-e730-687d-c5bc7f068907%22%2C%22optOut%22%3Afalse%7D%7D; _hjSession_1667000=eyJpZCI6ImNmMTY3ZjdhLWNlZDMtNDE0Yy1iYjQxLTJhNzNmNzFlNjlkZSIsImMiOjE3NTA2OTgyOTk0MzgsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjowLCJzcCI6MX0=; krg_crb=%7B%22v%22%3A%22eyJjbGllbnRJZCI6IjQ2YWI1ZGVjLWNhNmEtNDZjOS1iNjc3LWNmNGYzZDY2YmU2NyIsInRkSUQiOiJiNjBiNmE5YS05ZjJkLTRhYWMtYTQyMS1jNjc5ZGEyYTM3ZWEiLCJsZXhJZCI6ImU3NWJlMzU5LWZhZmYtZTczMC02ODdkLWM1YmM3ZjA2ODkwNyIsImt0Y0lkIjoiOGMyZjExZDEtYzMxZC0wNDQxLTViYWUtZjI4MmI2YjUyMDIzIiwiZXhwaXJlVGltZSI6MTc1MDc4NDc0NzQ5OSwibGFzdFN5bmNlZEF0IjoxNzUwNjk1MDU4MzcwLCJwYWdlVmlld0lkIjoiIiwicGFnZVZpZXdUaW1lc3RhbXAiOjE3NTA2OTgzNDc0OTEsInBhZ2VWaWV3VXJsIjoiaHR0cHM6Ly93d3cucmF0ZW15cHJvZmVzc29ycy5jb20vc2VhcmNoL3Byb2Zlc3NvcnMvMTEwMSIsInVzcCI6IjEtLS0ifQ%3D%3D%22%7D; _lr_retry_request=true; __gads=ID=0b9ddb10485796ad:T=1750694831:RT=1750700413:S=ALNI_MYvRnIcaHqeck8jelyxjNi_RKFsUg; __gpi=UID=0000104b48c16f7b:T=1750694831:RT=1750700413:S=ALNI_MbNO1qbyooHLJs71Ag7Ubs0lfT2DA; __eoi=ID=45438e83d624603d:T=1750694831:RT=1750700413:S=AA-AfjahG71eV97CY_boztorIfUr; connectId=%7B%22puid%22%3A%22739d06716ff0187f94c3bea60fc3209dfc1827bebe2eef0671afb1a0668580da%22%2C%22vmuid%22%3A%22Mrbyzk7pZrKqtx07STYqVZpqS8pXwf-ksCsvpoS6JuHzbj4Uq_Zcl2yKlUUPJRzTcsHOQO4EPn60T6j9pS0eKA%22%2C%22connectid%22%3A%22Mrbyzk7pZrKqtx07STYqVZpqS8pXwf-ksCsvpoS6JuHzbj4Uq_Zcl2yKlUUPJRzTcsHOQO4EPn60T6j9pS0eKA%22%2C%22connectId%22%3A%22Mrbyzk7pZrKqtx07STYqVZpqS8pXwf-ksCsvpoS6JuHzbj4Uq_Zcl2yKlUUPJRzTcsHOQO4EPn60T6j9pS0eKA%22%2C%22ttl%22%3A86400000%2C%22lastSynced%22%3A1750694870913%2C%22lastUsed%22%3A1750700415140%7D; AWSALB=7EvINk0G+8xswnmFii5KzSV/xICo6GRrDzIUK+KaHe3L4pn2hoAVqwnjTQ8UVTKsa4oQbeOOQhVY7l5Zg9peh4kNvDSmpzf1KDO2sUsgq50ytG+kfckw8iDmjxiwTaIs/AeEqHUbh2dsdkh/+ugtdGBOkyDwBPv8JFQy/DNJeNiOEn8zVIcIp7m6AsJ/RQ==; AWSALBCORS=7EvINk0G+8xswnmFii5KzSV/xICo6GRrDzIUK+KaHe3L4pn2hoAVqwnjTQ8UVTKsa4oQbeOOQhVY7l5Zg9peh4kNvDSmpzf1KDO2sUsgq50ytG+kfckw8iDmjxiwTaIs/AeEqHUbh2dsdkh/+ugtdGBOkyDwBPv8JFQy/DNJeNiOEn8zVIcIp7m6AsJ/RQ==; _ga_FVWZ0RM4DH=GS2.1.s1750698345$o2$g1$t1750700419$j53$l0$h0; cto_bundle=590gdF9HV1hRRjJicERUSU4xT3B6TnBzaGt2dEhzaHlTQnBrd29iZnRxRVA3RDRUSno0WnlQTW9DaHN2R2NVJTJCSENUUyUyQldQNDhvY2VweGJ1Z0pORGpuRTFkOG1lTGUxMnA2c0R4UXg3MVZBamJtVjU0Z1hxUzJ2SVlienBlelJpZEFiNlY4MFc0R0c5UWdPOHhBQ2hJdHNKaHolMkJrTmxRRUVwa0lwRFlqQkVPZ0FkZjglM0Q; cto_bidid=hkR_519ONW5vaW1oWkJ3aTJKSVQ0NlJvQjBweElYciUyQnUxcFI0WW5BZUwlMkJnbEVYaGIybEFydlZOWjBabGpJSFNmcng3Mk8yY0wxamJJa0Z1JTJGc0wxRlZIUVFmdDlPSWszeldFaFFLSDMzcGd2V0J1JTJGV0lpU0VGU3F6SVkzVmRuVnN2JTJCUGg; _pubcid_cst=VyxHLMwsHQ%3D%3D; _awl=2.1750700420.5-46549bce9c7f7d3e05cc5686eddc5dca-6763652d75732d6561737431-0; FCNEC=%5B%5B%22AKsRol_pmdJ3QyEVICWtGD7xDLftySVFqlEaMwN49wafLju2doiNci92uV0UZnW2xqJmkkDDr8O0a1Hzt3RlhJh2Q3HL2YDA_Gs1ctSq1Pfyi_p2mhm7apEqB2iUAWfQJtZvm660bDbG06wqv2Ug5buq7LKEJ71w0g%3D%3D%22%5D%5D; _ga_WET17VWCJ3=GS2.1.s1750698300$o2$g1$t1750700428$j45$l0$h0"
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"

    def start_requests(self):
        # Start with the first page (no cursor)
        variables = {
            "count": 50,  # Fetch 50 at a time for efficiency
            "cursor": None,  # Will be converted to null in JSON
            "query": {"text": "", "schoolID": self.school_id, "fallback": True}
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
            "Cookie": self.cookie_header,
            "Origin": "https://www.ratemyprofessors.com",
            "Referer": "https://www.ratemyprofessors.com/search/professors/1101?q=*",
            "Accept": "*/*",
        }
        yield scrapy.Request(
            url=self.graphql_url,
            method="POST",
            headers=headers,
            body=json.dumps({"query": self.graphql_query, "variables": variables}),
            callback=self.parse_professors,
            cb_kwargs={"variables": variables, "headers": headers}
        )

    def parse_professors(self, response, variables, headers):
        self.logger.info(f"GraphQL response: {response.text[:1000]}")  # Print first 1000 chars
        try:
            data = response.json()
        except Exception as e:
            self.logger.error(f"Failed to parse JSON: {e}\nRaw response: {response.text}")
            return
        if not data.get("data") or not data["data"].get("search"):
            self.logger.error(f"Unexpected response structure: {data}")
            return
        teachers = data["data"]["search"]["teachers"]
        for edge in teachers["edges"]:
            node = edge["node"]
            # Build full name
            name_parts = [node.get("firstName", ""), node.get("lastName", "")]
            name = " ".join([part for part in name_parts if part]).strip()
            professor = {
                "name": name,
                "avg_rating": node.get("avgRating"),
                "would_take_again": node.get("wouldTakeAgainPercent"),
                "difficulty": node.get("avgDifficulty"),
                "profile_url": f"https://www.ratemyprofessors.com/professor/{node['legacyId']}",
                "reviews": [],
            }
            # Go to the professor's detail page for reviews
            yield scrapy.Request(
                url=professor["profile_url"],
                callback=self.parse_professor,
                cb_kwargs={"professor": professor}
            )
        # Pagination
        page_info = teachers["pageInfo"]
        if page_info["hasNextPage"]:
            variables["cursor"] = page_info["endCursor"]
            yield scrapy.Request(
                url=self.graphql_url,
                method="POST",
                headers=headers,
                body=json.dumps({"query": self.graphql_query, "variables": variables}),
                callback=self.parse_professors,
                cb_kwargs={"variables": variables, "headers": headers}
            )

    def parse_professor(self, response, professor):
        # Extract department information from professor profile
        department = response.css('a.TeacherDepartment__StyledDepartmentLink-fl79e8-0 b::text').get()
        if department:
            professor['department'] = department.replace(' department', '').strip()
        
        # Extract reviews on this page
        review_blocks = response.css('div.Rating__RatingBody-sc-1rhvpxz-0')
        for block in review_blocks:
            review = {}
            # Review text and date
            review['text'] = block.css('div.Comments__StyledComments-dzzyvm-0::text').get()
            review['date'] = block.css('div.TimeStamp__StyledTimeStamp-sc-9q2r30-0::text').get()
            # Course code (robust)
            course_codes = block.xpath('.//div[contains(@class, "RatingHeader__StyledClass-sc-1dlkqw1-3")]/text()').getall()
            course_code = next((c.strip() for c in course_codes if c.strip()), None)
            if course_code:
                review['course_code'] = course_code
            # Quality and Difficulty
            quality = block.xpath('.//div[contains(text(), "Quality")]/following-sibling::div[1]/text()').get()
            if quality:
                review['quality'] = quality.strip()
            difficulty = block.xpath('.//div[contains(text(), "Difficulty")]/following-sibling::div[1]/text()').get()
            if difficulty:
                review['difficulty'] = difficulty.strip()
            # Attendance, Textbook, and other metadata
            meta_items = block.css('div.MetaItem__StyledMetaItem-y0ixml-0')
            for item in meta_items:
                label = item.xpath('text()').get()
                value = item.css('span::text').get()
                if label and value:
                    label = label.strip().replace(':', '')
                    value = value.strip()
                    if label == 'For Credit':
                        review['for_credit'] = value
                    elif label == 'Attendance':
                        review['attendance'] = value
                    elif label == 'Would Take Again':
                        review['would_take_again'] = value
                    elif label == 'Grade':
                        review['grade'] = value
                    elif label == 'Textbook':
                        review['textbook'] = value
                    elif label == 'Online Class':
                        review['online_class'] = value
            # Tags
            review['tags'] = block.css('span.Tag-bs9vf4-0::text').getall()
            professor['reviews'].append(review)
        # Check for a "next page" button for reviews
        next_page = response.css('button[aria-label="Next Page"]')
        if next_page:
            match = re.search(r'/professor/(\d+)', response.url)
            if match:
                prof_id = match.group(1)
                next_page_num = response.meta.get('page', 1) + 1
                next_url = f"https://www.ratemyprofessors.com/professor/{prof_id}?page={next_page_num}"
                yield scrapy.Request(
                    next_url,
                    callback=self.parse_professor,
                    cb_kwargs={"professor": professor},
                    meta={"page": next_page_num}
                )
            else:
                yield professor
        else:
            yield professor 