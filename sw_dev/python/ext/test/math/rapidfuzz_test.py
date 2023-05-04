#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rapidfuzz

# REF [site] >> https://github.com/maxbachmann/RapidFuzz
def simple_example():
	# Scorers.

	# Simple ratio.
	score = rapidfuzz.fuzz.ratio("this is a test", "this is a test!")
	print(f"{score=}")

	# Partial ratio.
	score = rapidfuzz.fuzz.partial_ratio("this is a test", "this is a test!")
	print(f"{score=}")

	# Token sort ratio.
	score = rapidfuzz.fuzz.ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
	print(f"{score=}")
	score = rapidfuzz.fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
	print(f"{score=}")

	# Token set ratio.
	score = rapidfuzz.fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
	print(f"{score=}")
	score = rapidfuzz.fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
	print(f"{score=}")

	#-----
	# Weighted ratio.
	score = rapidfuzz.fuzz.WRatio("this is a test", "this is a new test!!!", processor=None)
	print(f"{score=}")

	# Removing non alpha numeric characters("!") from the string.
	score = rapidfuzz.fuzz.WRatio("this is a test", "this is a new test!!!", processor=rapidfuzz.utils.default_process)  # Here "this is a new test!!!" is converted to "this is a new test".
	print(f"{score=}")
	score = rapidfuzz.fuzz.WRatio("this is a test", "this is a new test")
	print(f"{score=}")

	# Converting string to lower case.
	score = rapidfuzz.fuzz.WRatio("this is a word", "THIS IS A WORD", processor=None)
	print(f"{score=}")
	score = rapidfuzz.fuzz.WRatio("this is a word", "THIS IS A WORD", processor=rapidfuzz.utils.default_process)  # Here "THIS IS A WORD" is converted to "this is a word".
	print(f"{score=}")

	#-----
	# Quick ratio.
	score = rapidfuzz.fuzz.QRatio("this is a test", "this is a new test!!!", processor=None)
	print(f"{score=}")

	# Removing non alpha numeric characters("!") from the string.
	score = rapidfuzz.fuzz.QRatio("this is a test", "this is a new test!!!", processor=rapidfuzz.utils.default_process)
	print(f"{score=}")
	score = rapidfuzz.fuzz.QRatio("this is a test", "this is a new test")
	print(f"{score=}")

	# Converting string to lower case.
	score = rapidfuzz.fuzz.QRatio("this is a word", "THIS IS A WORD", processor=None)
	print(f"{score=}")
	score = rapidfuzz.fuzz.QRatio("this is a word", "THIS IS A WORD", processor=rapidfuzz.utils.default_process)
	print(f"{score=}")

	#--------------------
	# Process.

	choices = ["Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys"]

	results = rapidfuzz.process.extract("new york jets", choices, scorer=rapidfuzz.fuzz.WRatio, limit=2, processor=None)
	print(f"{results=}")
	result = rapidfuzz.process.extractOne("cowboys", choices, scorer=rapidfuzz.fuzz.WRatio, processor=None)
	print(f"{result=}")

	# With preprocessing.
	results = rapidfuzz.process.extract("new york jets", choices, scorer=rapidfuzz.fuzz.WRatio, limit=2, processor=rapidfuzz.utils.default_process)
	print(f"{results=}")
	result = rapidfuzz.process.extractOne("cowboys", choices, scorer=rapidfuzz.fuzz.WRatio, processor=rapidfuzz.utils.default_process)
	print(f"{result=}")

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
