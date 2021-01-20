#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://scrapy.org/

import scrapy

class BlogSpider(scrapy.Spider):
	name = 'blogspider'
	start_urls = ['https://blog.scrapinghub.com']

	def parse(self, response):
		for title in response.css('.post-header>h2'):
			yield {'title': title.css('a ::text').get()}

		for next_page in response.css('a.next-posts-link'):
			yield response.follow(next_page, self.parse)

#--------------------------------------------------------------------

# Usage:
#	scrapy runspider scrapy_test.py

if '__main__' == __name__:
	main()
