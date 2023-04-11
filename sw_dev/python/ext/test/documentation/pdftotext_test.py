#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pdftotext

# REF [site] >> https://github.com/jalan/pdftotext
def simple_example():
	pdf_file_path = "/path/to/example.pdf"

	# Load a PDF.
	try:
		with open(pdf_file_path, "rb") as fd:
			# raw: If True, page text is output in the order it appears in the content stream.
			# physical: If True, page text is output in the order it appears on the page, regardless of columns or other layout features.
			#pdf = pdftotext.PDF(fd, password="", raw=False, physical=False)
			pdf = pdftotext.PDF(fd)
			#pdf = pdftotext.PDF(fd, password="password")
	except UnicodeDecodeError as ex:
		print(f"Unicode decode error in {pdf_file_path}: {ex}.")
		return
	except FileNotFoundError as ex:
		print(f"File not found, {pdf_file_path}: {ex}.")
		return

	# How many pages?
	print(f"#page = {len(pdf)}.")

	# Iterate over all the pages.
	for idx, page in enumerate(pdf):
		print(f"Page #{idx} --------------------")
		print(page)

	# Read some individual pages.
	print("Page #0 --------------------")
	print(pdf[0])
	print("Page #1 --------------------")
	print(pdf[1])

	# Read all the text into one string.
	print("All pages --------------------")
	print("\n\n".join(pdf))

def main():
	simple_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
