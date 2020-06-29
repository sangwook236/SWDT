#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/lumyjuwon/KoreaNewsCrawler
def naver_news_crawling_example():
	from korea_news_crawler.articlecrawler import ArticleCrawler

	crawler = ArticleCrawler()  
	crawler.set_category('정치', 'IT과학', 'economy')
	#crawler.set_category('politics', 'IT_science', 'economy')  
	crawler.set_date_range(2017, 1, 2018, 4)  
	crawler.start()

	# Output: CSV file.
	#	Column: 기사 날짜, 기사 카테고리, 언론사, 기사 제목, 기사 본문, 기사 주소.

def main():
	naver_news_crawling_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
