#!/usr/bin/env python
# -*- coding: UTF-8 -*-

def google_search_example():
	from googlesearch import search

	query = "Medium.com"

	for url in search(query):
		print(url)

def main():
	google_search_example()

#--------------------------------------------------------------------
# Usage:
#	pip install google

if "__main__" == __name__:
	main()
