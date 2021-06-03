#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import jellyfish

# REF [site] >> https://github.com/jamesturk/jellyfish
def simple_example():
	# String comparison.
	str1, str2 = u'jellyfish', u'smellyfish'

	print("jellyfish.levenshtein_distance({}, {}) = {}.".format(str1, str2, jellyfish.levenshtein_distance(str1, str2)))
	print("jellyfish.damerau_levenshtein_distance({}, {}) = {}.".format(str1, str2, jellyfish.damerau_levenshtein_distance(str1, str2)))
	print("jellyfish.hamming_distance({}, {}) = {}.".format(str1, str2, jellyfish.hamming_distance(str1, str2)))
	print("jellyfish.jaro_distance({}, {}) = {}.".format(str1, str2, jellyfish.jaro_distance(str1, str2)))
	print("jellyfish.jaro_similarity({}, {}) = {}.".format(str1, str2, jellyfish.jaro_similarity(str1, str2)))
	print("jellyfish.jaro_winkler({}, {}) = {}.".format(str1, str2, jellyfish.jaro_winkler(str1, str2)))
	print("jellyfish.jaro_winkler_similarity({}, {}) = {}.".format(str1, str2, jellyfish.jaro_winkler_similarity(str1, str2)))
	print("jellyfish.match_rating_comparison({}, {}) = {}.".format(str1, str2, jellyfish.match_rating_comparison(str1, str2)))

	#--------------------
	# Phonetic encoding.
	ss = u'Jellyfish'

	print("jellyfish.metaphone({}) = {}.".format(ss, jellyfish.metaphone(ss)))
	print("jellyfish.soundex({}) = {}.".format(ss, jellyfish.soundex(ss)))
	print("jellyfish.nysiis({}) = {}.".format(ss, jellyfish.nysiis(ss)))
	print("jellyfish.match_rating_codex({}) = {}.".format(ss, jellyfish.match_rating_codex(ss)))

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
