#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://docs.python.org/3/library/re.html
#	https://docs.python.org/3/howto/regex.html

import re

def basic_operation():
	r"""
	# Special sequence.
	\number \A \b \B \d \D \s \S \w \W \Z

	# Standard escape.
	\a \b \f \n \N \r \t \u \U \v \x \\

	# Flag.
	re.A, re.ASCII
	re.I, re.IGNORECASE
	re.L, re.LOCALE
	re.M, re.MULTILINE
	re.S, re.DOTALL
	re.U, re.UNICODE
	re.X, re.VERBOSE

	re.DEBUG

	re.search(pattern, string, flags=0)
		Scan through string looking for the first location where the regular expression pattern produces a match.
	re.match(pattern, string, flags=0)
		If zero or more characters at the beginning of string match the regular expression pattern.
	re.fullmatch(pattern, string, flags=0)
		If the whole string matches the regular expression pattern.
	re.split(pattern, string, maxsplit=0, flags=0)
	re.findall(pattern, string, flags=0)
	re.finditer(pattern, string, flags=0)
	re.sub(pattern, repl, string, count=0, flags=0)
	re.subn(pattern, repl, string, count=0, flags=0)

	re.escape(pattern)

	re.purge()
	"""

	#--------------------
	# Search.

	# *, +, ?.
	#	The '*', '+', and '?' qualifiers are all greedy; they match as much text as possible.
	#	If the RE <.*> is matched against '<a> b <c>', it will match the entire string, and not just '<a>'.
	# *?, +?, ??.
	#	Adding ? after the qualifier makes it perform the match in non-greedy or minimal fashion; as few characters as possible will be matched.
	#	Using the RE <.*?> will match only '<a>' against '<a> b <c>'.

	re.search(r'''['"].*['"]''', '''ab'cd'ef'gh'ij"kl"mn'op"qr"st'uv"wx'yz'AB"CD''')  # Result: '\'cd\'ef\'gh\'ij"kl"mn\'op"qr"st\'uv"wx\'yz\'AB"'.
	re.search(r'''['"].*?['"]''', '''ab'cd'ef'gh'ij"kl"mn'op"qr"st'uv"wx'yz'AB"CD''')  # Result: "'cd'".

	# (...): Group.
	# (?P<name>...): Named group.
	# (?P=name): Backreference to a named group.

	re.search(r'''(?P<quote>['"]).*(?P=quote)''', '''ab'cd'ef'gh'ij"kl"mn'op"qr"st'uv"wx'yz'AB"CD''')  # Result: '\'cd\'ef\'gh\'ij"kl"mn\'op"qr"st\'uv"wx\'yz\''.
	re.search(r'''(?P<quote>['"]).*?(?P=quote)''', '''ab'cd'ef'gh'ij"kl"mn'op"qr"st'uv"wx'yz'AB"CD''')  # Result: "'cd'".
	re.search(r'''(?P<asterisk>\*).*?(?P=asterisk)|(?P<quote>['"]).*?(?P=quote)''', '''ab'cd'ef'gh'ij"kl"mn*op*qr'st"uv"wx'yz*AB*CD"EF'GH'IJ"KL*MN*OPQRSTUVWXYZ''')  # Result: "'cd'".

	# (?=...): Lookahead assertion.
	# (?!...): Negative lookahead assertion.
	# (?<=...): Positive lookbehind assertion.
	# (?<!...): Negative lookbehind assertion.

	# (?!...): Negative lookahead assertion.
	re.search(r'(?!ABC)\w*', 'Aabcde')  # Matched.
	re.search(r'(?!ABC)\w*', 'Babcde')  # Matched.
	re.search(r'(?!ABC)\w*', 'Cabcde')  # Matched.
	re.search(r'(?!ABC)\w*', 'ABabcde')  # Matched.
	re.search(r'(?!ABC)\w*', 'BCabcde')  # Matched.
	re.search(r'(?!ABC)\w*', 'ABCabcde')  # Unmatched.

	#--------------------
	# Match.

	# [^...]: Complementation of a set of characters.
	# The first character.
	re.match(r'[^A]\w*', 'abcde')  # Matched.
	re.match(r'[^A]\w*', 'Babcde')  # Matched.
	re.match(r'[^A]\w*', 'Aabcde')  # Unmatched.
	re.match(r'[^ABC]\w*', 'abcde')  # Matched.
	re.match(r'[^ABC]\w*', 'aAabcde')  # Matched.
	re.match(r'[^ABC]\w*', 'Aabcde')  # Unmatched.
	re.match(r'[^ABC]\w*', 'Babcde')  # Unmatched.
	re.match(r'[^ABC]\w*', 'Cabcde')  # Unmatched.
	# The second character.
	re.match(r'\w[^A]\w*', 'abcde')  # Matched.
	re.match(r'\w[^A]\w*', 'aBabcde')  # Matched.
	re.match(r'\w[^A]\w*', 'aAabcde')  # Unmatched.
	# The first and second characters.
	re.match(r'[^A][^B]\w*', 'abcde')  # Matched.
	re.match(r'[^A][^B]\w*', 'Babcde')  # Matched.
	re.match(r'[^A][^B]\w*', 'Aabcde')  # Unmatched.
	re.match(r'[^A][^B]\w*', 'aBabcde')  # Unmatched.

	#--------------------
	# Split.

	re.split(r'\W+', 'Words, words, words.')
	re.split(r'(\W+)', 'Words, words, words.')
	re.split(r'\W+', 'Words, words, words.', 1)
	re.split('[a-f]+', '0a3B9', flags=re.IGNORECASE)
	re.split(r'(\W+)', '...words, words...')

	#--------------------
	# Find.

	re.findall(r'''['"].*?['"]''', '''ab'cd'ef'gh'ij"kl"mn'op"qr"st'uv"wx'yz'AB"CD''')  # Result: ["'cd'", "'gh'", '"kl"', '\'op"', '"st\'', '"wx\'', '\'AB"'].
	re.findall(r'''(?P<quote>['"]).*?(?P=quote)''', '''ab'cd'ef'gh'ij"kl"mn'op"qr"st'uv"wx'yz'AB"CD''')  # Result: ["'", "'", '"', "'", '"'].

	re.findall(r'''['"].*?['"]|\*.*?\*''', '''ab'cd'ef'gh'ij"kl"mn*op*qr'st"uv"wx'yz*AB*CD"EF'GH'IJ"KL*MN*OPQRSTUVWXYZ''')  # Result: ["'cd'", "'gh'", '"kl"', '*op*', '\'st"', '"wx\'', '*AB*', '"EF\'', '\'IJ"', '*MN*'].
	re.findall(r'''(?P<quote>['"]).*?(?P=quote)|(?P<asterisk>\*).*?(?P=asterisk)''', '''ab'cd'ef'gh'ij"kl"mn*op*qr'st"uv"wx'yz*AB*CD"EF'GH'IJ"KL*MN*OPQRSTUVWXYZ''')  # Result: [("'", ''), ("'", ''), ('"', ''), ('', '*'), ("'", ''), ('', '*'), ('"', ''), ('', '*')].
	re.findall(r'''(?P<asterisk>\*).*?(?P=asterisk)|(?P<quote>['"]).*?(?P=quote)''', '''ab'cd'ef'gh'ij"kl"mn*op*qr'st"uv"wx'yz*AB*CD"EF'GH'IJ"KL*MN*OPQRSTUVWXYZ''')  # Result: [('', "'"), ('', "'"), ('', '"'), ('*', ''), ('', "'"), ('*', ''), ('', '"'), ('*', '')].

	#--------------------
	# Substitute.

	def dash_repl(match):
		if match.group(0) == '-': return ' '  # The entire match.
		else: return '-'
	re.sub('-{1,2}', '-', 'pro----gram-files')  # Result: "pro--gram-files".
	re.sub('-{1,2}', dash_repl, 'pro----gram-files')  # Result: "pro--gram files".
	re.sub(r'\sAND\s', ' & ', 'Baked Beans And Spam', flags=re.IGNORECASE)  # Result: "Baked Beans & Spam".

	re.subn('-{1,2}', dash_repl, 'pro----gram-files')  # Result: "('pro--gram files', 3)".
	re.subn(r'\sAND\s', ' & ', 'Baked Beans And Spam', flags=re.IGNORECASE)  # Result: "('Baked Beans & Spam', 1)".

	#--------------------
	re.escape('http://www.python.org')  # Result: "http://www\\.python\\.org".

	re.purge()  # Clear the regular expression cache.

	#--------------------
	try:
		re.compile('[a-z+')
	except re.error as ex:
		print('re.error: {}.'.format(ex))

# Compiled regular expression object.
def pattern_object_example():
	"""
	re.compile(pattern, flags=0)

	Pattern.search(string, pos, endpos)
	Pattern.match(string, pos, endpos)
	Pattern.fullmatch(string, pos, endpos)
	Pattern.split(string, maxsplit=0)
	Pattern.findall(string, pos, endpos)
	Pattern.finditer(string, pos, endpos)
	Pattern.sub(repl, string, count=0)
	Pattern.subn(repl, string, count=0)

	Pattern.flags
	Pattern.groups
	Pattern.groupindex
	Pattern.pattern
	"""

	pattern = re.compile('[a-z]+')
	#pattern = re.compile('[a-z]+', re.IGNORECASE)
	#pattern = re.compile(r'(\w+) (\w+)')
	#pattern = re.compile(r'\d+\.\d*', re.X)

	print('Pattern = {}.'.format(pattern.pattern))
	print('\tFlag = {}.'.format(pattern.flags))
	print('\t#groups = {}.'.format(pattern.groups))
	print('\tGroup index = {}.'.format(pattern.groupindex))

	#--------------------
	# Search.

	pattern = re.compile('d')
	pattern.search('dog')  # Match at index 0.
	pattern.search('dog', 1)  # No match; search doesn't include the 'd'.

	#--------------------
	# Match.

	pattern = re.compile('o')
	pattern.match('dog')  # No match as 'o' is not at the start of 'dog'.
	pattern.match('dog', 1)  # Match as 'o' is the 2nd character of 'dog'.

	pattern = re.compile('o[gh]')
	pattern.fullmatch('dog')  # No match as "o" is not at the start of 'dog'.
	pattern.fullmatch('ogre')  # No match as not the full string matches.
	pattern.fullmatch('doggie', 1, 3)  # Matches within given limits.

	#--------------------
	# Split.

	#--------------------
	# Find.

	pattern = re.compile(r'\d+')
	print(pattern.findall('12 drummers drumming, 11 pipers piping, 10 lords a-leaping'))

	iterator = pattern.finditer('12 drummers drumming, 11 ... 10 ...')
	for match in iterator:
		print(match.span())

	#--------------------
	# Substitute.

# Match object.
def match_object_example():
	"""
	Match.expand()
	Match.group()
	Match.groups()
	Match.groupdict()
	Match.start()
	Match.end()
	Match.span()

	Match.pos
	Match.endpos
	Match.lastindex
	Match.lastgroup
	Match.re
	Match.string
	"""

	pattern = re.compile('[a-z]+')

	print(pattern)
	print(pattern.match(''))

	match = pattern.match('tempo')
	if match:
		print(match)
		print(match.group())
		print(match.start(), match.end())
		print(match.span())
	else:
		print('No match')

	match = re.match(r'(\w+) (\w+)', 'Isaac Newton, physicist')
	match.group(0)  # The entire match.
	match.group(1)  # The first parenthesized subgroup.
	match.group(2)  # The second parenthesized subgroup.
	match.group(1, 2)  # Multiple arguments give us a tuple.

	match = re.match(r'(?P<first_name>\w+) (?P<last_name>\w+)', 'Malcolm Reynolds')
	match.group('first_name')
	match.group('last_name')
	match.group(1)
	match.group(2)

	# If a group matches multiple times, only the last match is accessible.
	match = re.match(r'(..)+', 'a1b2c3')  # Matches 3 times.
	match.group(1)  # Returns only the last match.

	match = re.match(r'(\d+)\.(\d+)', '24.1632')
	match.groups()

	match = re.match(r'(\d+)\.?(\d+)?', '24')
	match.groups()  # Second group defaults to None.
	match.groups('0')  # The second group defaults to '0'.

	match = re.match(r'(?P<first_name>\w+) (?P<last_name>\w+)', 'Malcolm Reynolds')
	match.groupdict()

# REF [site] >> Commonly used regular expressions.
#	https://digitalfortress.tech/tricks/top-15-commonly-used-regex/
def commonly_used_expressions():
	# Roman numerals.
	#	I: 1, V: 5, X: 10, L: 50, C: 100, D: 500, M: 1000.

	roman_numerals_correct = [
		'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
		'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
		'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
		'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx',
	]
	roman_numerals_incorrect = [
		'I.',
		'IIII',
		'IIV', 'VIIII', 'VV',
		'IIX','XIIII',
	]

	# REF [site] >> https://www.oreilly.com/library/view/regular-expressions-cookbook/9780596802837/ch06s09.html
	#roman_numeral_pattern = re.compile(r'^[MDCLXVI]+$', re.I)  # No validation.
	roman_numeral_pattern = re.compile(r'^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$', re.I)  # Strict.
	#roman_numeral_pattern = re.compile(r'^(?=[MDCLXVI])M*(C[MD]|D?C*)(X[CL]|L?X*)(I[XV]|V?I*)$', re.I)  # Flexible.
	#roman_numeral_pattern = re.compile(r'^(?=[MDCLXVI])M*D?C{0,4}L?X{0,4}V?I{0,4}$', re.I)  # Simple.
	# REF [site] >> https://stackoverflow.com/questions/267399/how-do-you-match-only-valid-roman-numerals-with-a-regular-expression
	#roman_numeral_pattern = re.compile(r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', re.I)

	for ss in roman_numerals_correct:
		match = roman_numeral_pattern.match(ss)
		if match is None:
			print('Roman numeral correct (unmatched): {}.'.format(ss))
		elif ss != match[0]:
			print('Roman numeral correct (partially matched): {} != {}.'.format(match[0], ss))

	for ss in roman_numerals_incorrect:
		#match = roman_numeral_pattern.match(ss)
		match = roman_numeral_pattern.fullmatch(ss)
		if match is not None:
			if ss == match[0]:
				print('Roman numeral incorrect (matched): {}.'.format(ss))
			else:
				print('Roman numeral incorrect (partially matched): {} != {}.'.format(match[0], ss))

	#--------------------
	# HTML tag.
	html_tag_pattern = re.compile(r'<\/?[\w\s]*>|<.+[\W]>')

def misc_expressions():
	# Author list in papers.
	paper_author_list_correct = [
		'M. M. N. Minkman*',
		'S. Y. Matthew Goh, Andrei Irimia, Carinna M. Torgerson and John D. Van Horn*',
		'A. Matthew Prina,1* Daisy Acosta,2 Isaac Acosta,3 Mariella Guerra,4 Yueqin Huang,5 A.T. Jotheeswaran,6 Ivonne Z. Jimenez-Velazquez,7 Zhaorui Liu,5 Juan J. Llibre Rodriguez,8 Aquiles Salas,9 Ana Luisa Sosa,3 Joseph D. Williams10 and Martin Prince1',
		'R. Wickström*',
		'R. S. Dow Neurobiology Laboratory, Portland OR, USA ',
		'R. Miller* ',
		'K. Tanaka1, R. F. S.-Galduróz2,3, L. T. B. Gobbi2 and J. C. F. Galduróz1,* ',
		'A. Siniscalchi1, L. Gallelli2,* and G. De Sarro2 ',
		'D. Moharregh-Khiabani1, R.A. Linker2, R. Gold2 and M. Stangel1,* ',
		'B. Di Benedetto1,2,* and R. Rupprecht1,2,* ',
		'P. Sáez-Briones1,* and A. Hernández 2 ',
		'R. Miller* ',
		'F. Ferrinia, C. Salioa, L. Lossia,b and A. Merighia,b,*',
		'C. A. Fontes-Ribeiro*, E. Marques , F. C. Pereira, A. P. Silva and T. R. A. Macedo ',
		'J. Law*, P. Shaw , K. Earland , M. Sheldon and M. Lee',
		'J. Packer, M. A. Hussain, S. H. A. Shah, and J. R. Srinivasan',
		'J. A. Ashindoitiang',
		'A. Harb and A. N. Pandya',
		'A. K. Singh and S. S. Narsipur',
		'F. Cremonesi,1 K. Anderson,2 and A. Lange-Consiglio1',
		'J. Gasteiner, T. Guggenberger, J. H¨ausler, and A. Steinwidder',
		'C. Pellicoro,1 R. Marsella,2 and K. Ahrens2',
		'M. Nalaka Munasinghe,1 Craig Stephen,2, 3 Preeni Abeynayake,1 and Indra S. Abeygunawardena1',
		'C. G. Donnelly,1 C. T. Quinn,2 S. G. Nielsen,3 and S. L. Raidal2',
		'V. Sejian, V. P. Maurya, K. Kumar, and S. M. K. Naqvi',
		'J. R. Walton1, 2',
		'U. Beinhoff,1 H. Tumani,2 and M. W. Riepe1, 3',
		'S. C. Dyall',
		'S. E. Mason,1 R. McShane,2 and C. W. Ritchie1',
		'V. Senanarong, N. Siwasariyanon, L. Washirutmangkur, N. Poungvarin, C. Ratanabunakit, N. Aoonkaew, and S. Udomphanthurak',
		'M. Mancuso, V. Calsolaro, D. Orsucci, C. Carlesi, A. Choub, S. Piazza, and G. Siciliano',
		'A. Anoop, Pradeep K. Singh, Reeba S. Jacob, and Samir K. Maji',
		'D. C. Beck,1 Honglin Jiang,2 and Liqing Zhang1',
		'N. Konijnendijk,1, 2, 3 D. A. Joyce,2, 4 H. D. J. Mrosso,2, 5 M. Egas,3 and O. Seehausen1, 2',
		'R. Craig Albertson,1 W. James Cooper,2 and Kenneth A. Mann3',
		'M. Pilar Francino1, 2',
		'S. Shawn McCafferty,1 Andrew Martin,2 and Eldredge Bermingham3',
		'C. Verity Bennett1 and Anjali Goswami1, 2',
		'A. Bintoudi, K. Natsis, and I. Tsitouridis',
		'A. E. Evans, C. M. Kelly, S. V. Precious, and A. E. Rosser',
		'R. E. Foulkes, G. Heard, T. Boyce, R. Skyrme, P. A. Holland, and C. A. Gateley',
		'F. Lee Tucker',
		'N. S. Wind and I. Holen',
		'S. Gabal and S. Talaat',
		'J. L. Schwartz, A. N. Shajahan, and R. Clarke',
		'M. de Cesare1, Nick E. Mavromatos1,2, Sarben Sarkar1,a',
		'A. Tufano, E. Cimino, M. N. D. Di Minno, P. Ieran `o, E. Marrone, A. Strazzullo, G. Di Minno, and A. M. Cerbone',
		'R. Constance Wiener,1, 2 Rouxin Zhang,1 and Anoop Shankar1',
		'M. Moerland, A. J. Kales, L. Schrier, M. G. J. van Dongen, D. Bradnock, and J. Burggraaf',
		'S. Bassoli, S. Borsari, C. Ferrari, F. Giusti, G. Pellacani, G. Ponti, and S. Seidenari',
		'C. Thomas Dow1, 2 and Jay L. E. Ellingson3',
		'C. Mihl,1 B. S. N. Alzand,1 and M. H. Winkens1, 2',
		'R. B. Singh, Fabien DeMeester, and Agnieska Wilczynska',
		'K. M. Kurian',
		'W. D. Salman, Mayuri Singh, and Z. Twaij',
		'R. A. Armstrong',
		'A. Lissat,1 M. M. Chao,2 and U. Kontny1',
		'E. E. Philip-Ephraim,1 K. I. Eyong,2 U. E. Williams,1 and R. P. Ephraim1',
		'G. Fabbrocini, C. Mazzella, F. Pastore, A. Monfrecola, M. C. Annunziata, M. C. Mauriello, V. D’Arco, C. Marasca, and V. De Vita',
		'C. van der Leest,1, 2 A. Amelink,3 R. J. van Klaveren,2 H. C. Hoogsteden,2 H. J. C. M. Sterenborg,3 and J. G. J. V. Aerts1, 2',
		'R. Lau and M. A. C. Pratt',
		'P. Sobana Piriya, P. Thirumalai Vasan, V. S. Padma, U. Vidhyadevi,', 'K. Archana, and S. John Vennison',
		'G. Fava and I. Lorenzini',
		'M. Shadab Siddiqui1 and Richard K. Sterling1, 2',
		'F. Oesch1 · E. Fabian2 · Robert Landsiedel2',
		'E. C. Cashman1 and M. J. Donnelly2',
		'M. E. Huth,1, 2 A. J. Ricci,1, 3 and A. G. Cheng1',
		'E. C. Cashman,1 Terence Farrell,2 and M. Shandilya1',
		'L. M. Paulson, C. J. MacArthur, K. B. Beaulieu, J. H. Brockman, and H. A. Milczuk',
		'B. A. Rogers, A. Sternheim, D. Backstein, O. Saﬁr, and A. E. Gross',
		'V. Palepu, M. Kodigudla, and V. K. Goel',
		'D. Kok,1 R. D. Donk,2 F. H. Wapstra,1 and A. G. Veldhuizen1',
		'C. Moran and C. Bolger',
		'R. G. Kavanagh, J. S. Butler, J. M. O’Byrne, and A. R. Poynton',
		'A. R. Memon and J. F. Quinlan',
		'Y. Okahisa1, M. Kodama1, M. Takaki1, T. Inada2,3, N. Uchimura2,4, M. Yamada2,5, N. Iwata2,6, M. Iyo2,7, I. Sora2,8, N. Ozaki2,9 and H. Ujike1,2,* ',
		'Y. Takamatsu1, H. Shiotsuki1,2, S. Kasai1, S. Sato2, T. Iwamura3, N. Hattori2 and K. Ikeda1,* ',
		'R. Rajkumar* and R. Mahesh ',
		'Y. Haginoa, Y. Takamatsua, H. Yamamotoa, T. Iwamurab, D. L. Murphyc, G. R. Uhld, ', 'I. Sorae and K. Ikedaa,* ',
		'H. Ujike1,2,*, M. Kishimoto1, Y. Okahisa1, M. Kodama1, M. Takaki1, T. Inada2,3, N. Uchimura2,4, M. Yamada2,5, N. Iwata2,6, M. Iyo2,7, I. Sora2,8 and N. Ozaki2,9 ',
		'E. Y. Jang1, M. Hwang2, S. S. Yoon1, J. R. Lee2, K. J. Kim1, H.-C. Kim3 and C. H. Yang1,2,* ',
		'K. Ohira1,* and M. Hayashi2 ',
		'E. R. Samuels and E. Szabadi*',
		'F. Gasparini* and W. Spooren ',
		'E. Yokobayashi1, H. Ujike1,2,*, T. Kotaka1, Y. Okahisa1, M. Takaki1, M. Kodama1, T. Inada2,3, N. Uchimura2,4, M. Yamada2,5, N. Iwata2,6, M. Iyo2,7, I. Sora2,8, N. Ozaki2,9 and S. Kuroda1 ',
		'D. Marmolino1,* and M. Manto1,2 ',
		'Y. Takamatsu1, H. Yamamoto1, Y. Hagino1, A. Markou2 and K. Ikeda1,* ',
		'H. Yamamotoa,*, Y. Takamatsua, K. Imaib, E. Kamegayaa, Y. Haginoa, M. Watanabec,  T. Yamamotoa,d, I. Soraa,e, H. Kogab and K. Ikedaa ',
		'T. Okochi1,*, T. Kishi1, M. Ikeda1, T. Kitajima1, Y. Kinoshita1, K. Kawashima1, T. Okumura1, T. Tsunoka1, Y. Fukuo1, T. Inada2,9, M. Yamada3,9, N. Uchimura4,9, M. Iyo5,9, I. Sora6,9,  N. Ozaki7,9, H. Ujike8,9 and N. Iwata1,9 ',
		'D. Cervia* and G. Casini ',
		'N. Etheridgea, R. D. Mayfieldb, R. A. Harrisb and P. R. Dodda,* ',
		'C. Venâncioa,b, A. Magalhãesa, L. Antunesa,b and T. Summaviellea,* ',
		'Y. Okahisa1, M. Kodama1, M. Takaki1, T. Inada2,3, N. Uchimura2,4, M. Yamada2,5, N. Iwata2,6,  M. Iyo2,7, I. Sora2,8, N. Ozaki2,9 and H. Ujike1,2,* ',
		'M. Melis* and M. Pistis*',
		'G. Consiglieri,1 L. Leon-Chi,2 and R. S. Newfield1,2',
		'R. P. Treviño,1 T. H. Pham,2 and S. L. Edelstein2',
		'G. P. Stathopoulos1 and T. Boulikas2',
		'A. El-Ansary and S. Al-Daihan',
		'M. Marin-Kuan, V. Ehrlich, T. Delatour, C. Cavin, and B. Schilter',
		'S. S. Agrawal and R. S. Ray',
		'M. Lundberg,1 A. Grimby-Ekman,2 J. Verbunt,3 and M. J. Simmonds4, 5',
		'S. Stevens Negus,1 Ember M. Morrissey,1 John E. Folk,2 and Kenner C. Rice2',
		'A. K. Goel1, 2 and S. C. Jiang1',
		'K. Kealy Peak,1 Kathleen E. Duncan,2 Vicki A. Luna,1 Debra S. King,1 Peter J. McCarthy,3 and Andrew C. Cannons1',
		'H. Hassan1, 2 and M. Shorman3',
		'B. Segatore, D. Setacci, F. Bennato, R. Cardigno, G. Amicosante, and R. Iorio',
		'A. Mark Ibekwe,1 Sharon K. Papiernik,2 Catherine M. Grieve,1 and Ching-Hong Yang3',
	]
	paper_author_list_incorrect = [
		'M. M. N. Minkman,',
		'M. M. N. Minkman*,',
		'S. Y. Matthew Goh, Andrei Irimia, Carinna M. Torgerson and, John D. Van Horn*',
		'H. Hassan1, 2 and M. And Shorman3',
	]

	#paper_author_list_pattern = re.compile(r'((\w+\s+)*([A-Z]\.\s*)*([-\w]+\s*)+(,|,\s*(\d+|\*)\s*|(\d+|\*)\s*,\s*)?)+')
	paper_author_list_pattern = re.compile(r'(([Aa][Nn][Dd]\s+)?(\w+\s+)*([A-Z]\.\s*)*((?![Aa][Nn][Dd]\s+)[-\w]+\s*)+(,(\s*\*)?\s*|\*(\s*,)?\s*|,\s*(\d+|\*)\s*|(\d+|\*)(\s*,)?\s*)*)+')

	print('Paper author list pattern:')
	print('\tPattern = {}.'.format(paper_author_list_pattern.pattern))
	print('\tFlag = {}.'.format(paper_author_list_pattern.flags))
	print('\t#groups = {}.'.format(paper_author_list_pattern.groups))
	print('\tGroup index = {}.'.format(paper_author_list_pattern.groupindex))

	for ss in paper_author_list_correct:
		match = paper_author_list_pattern.match(ss)
		if match is None:
			print('Paper author list correct (unmatched): {}.'.format(ss))
		elif ss != match[0]:
			print('Paper author list correct (partially matched): {} != {}.'.format(match[0], ss))
	for ss in paper_author_list_incorrect:
		match = paper_author_list_pattern.match(ss)
		if match is not None:
			if ss == match[0]:
				print('Paper author list incorrect (matched): {}.'.format(ss))
			else:
				print('Paper author list incorrect (partially matched): {} != {}.'.format(match[0], ss))

def page_object_example():
	# Basic pattern.
	#re_pattern = re.compile(r'([0-9]+\.)*[0-9]+\.?')  # 1, 2., 1.2, 2.3., ...
	re_pattern = re.compile(r'(\d+\.)*\d+\.?')  # 1, 2., 1.2, 2.3., ...
	re_pattern = re.compile(r'[A-Za-z]\d*\.')  # A. B1., c2., ...
	# NOTE [info] >> As the target string is scanned, REs separated by '|' are tried from left to right.
	re_pattern = re.compile(r'([A-Za-z]\d*\.)?(\d+\.)*\d+\.?|[A-Za-z]\d*\.')
	re_pattern = re.compile(r'(([A-Za-z]\d*\.)?(\d+\.)*\d+\.?|[A-Za-z]\d*\.)\s+.+')

	#--------------------
	numberings_correct = [
		'3.', '3)', '(3)',
		'12345.', '12345)', '(12345)',
		'1.2', '1.2.', '1.2)', '(1.2)',
		'1.2.3.4.5', '1.2.3.4.5.', '1.2.3.4.5)', '(1.2.3.4.5)',
		'b.', 'b)', '(b)',
		'b.2', 'b.2.', 'b.2)', '(b.2)',
		'b137.', 'b137)', '(b137)',
		'b.1.2', 'b.1.2.', 'b.1.2)', '(b.1.2)',
		'b1.2', 'b1.2.', 'b1.2)', '(b1.2)',
		'b.1.2.3.4.5', 'b.1.2.3.4.5.', 'b.1.2.3.4.5)', '(b.1.2.3.4.5)',
		'b1.2.3.4.5', 'b1.2.3.4.5.', 'b1.2.3.4.5)', '(b1.2.3.4.5)',
	]
	numberings_incorrect = [
		'3', '12', '12345',
		'.3', '.12', '.12345',
		'b', 'b2', '2b', 'ab',
		'b.a1.2', 'b.a1.2.',
		'b.a.1.2', 'b.a.1.2.',
		'ab.1.2.3.4.5', 'ab.1.2.3.4.5.',
		'2b.1.2', '2b.1.2.',
		'2b.1.2.3.4.5', '2b.1.2.3.4.5.',
		'.b.1.2', '.b.1.2.',
		'.b.1.2.3.4.5', '.b.1.2.3.4.5.',
	]

	#--------------------
	# For numbering.
	if False:
		#re_pattern = re.compile(r'\(?([A-Za-z]\d*\.)?(\d+\.)*\d+\)|([A-Za-z]\d*\.)?(\d+\.)*\d+\.?|\(?[A-Za-z]\d*\)|[A-Za-z]\d*\.')
		re_pattern = re.compile(r'\(?([A-Za-z]\d*\.)?(\d+\.)*\d+\)|[A-Za-z]\d*\.(\d+\.)*\d+\.?|(\d+\.)+\d+\.?|\d+\.|\(?[A-Za-z]\d*\)|[A-Za-z]\d*\.')

		for ss in numberings_correct:
			match = re_pattern.match(ss)
			if match is None:
				print('Numbering correct (unmatched): {}.'.format(ss))
			elif ss != match[0]:
				print('Numbering correct (partially matched): {} != {}.'.format(match[0], ss))
			#else:
			#z	print('Numbering correct (matched): {}.'.format(ss))

		for ss in numberings_incorrect:
			#match = re_pattern.match(ss)
			match = re_pattern.fullmatch(ss)
			if match is not None:
				if ss == match[0]:
					print('Numbering incorrect (matched): {}.'.format(ss))
				else:
					print('Numbering incorrect (partially matched): {} != {}.'.format(match[0], ss))

	#--------------------
	# For heading.
	if True:
		#re_pattern = re.compile(r'(\(?([A-Za-z]\d*\.)?(\d+\.)*\d+\)|([A-Za-z]\d*\.)?(\d+\.)*\d+\.?|\(?[A-Za-z]\d*\)|[A-Za-z]\d*\.)\s+.+')
		re_pattern = re.compile(r'(\(?([A-Za-z]\d*\.)?(\d+\.)*\d+\)|[A-Za-z]\d*\.(\d+\.)*\d+\.?|(\d+\.)+\d+\.?|\d+\.|\(?[A-Za-z]\d*\)|[A-Za-z]\d*\.)\s+.+')

		numbered_heading_formats = [
			'{} Introduction',
			'{} Introduction.',
			'{} Experimental results',
			'{} Experimental results.',
			'{} Experimental results!',
			'{} Introduction. Experimental results',
			'{} Introduction. Experimental results.',
		]

		for head_fmt in numbered_heading_formats:
			for numbering in numberings_correct:
				ss = head_fmt.format(numbering)

				match = re_pattern.match(ss)
				if match is None:
					print('Numbered heading correct (unmatched): {}.'.format(ss))
				elif ss != match[0]:
					print('Numbered heading correct (partially matched): {} != {}.'.format(match[0], ss))

			for numbering in numberings_incorrect:
				ss = head_fmt.format(numbering)

				#match = re_pattern.match(ss)
				match = re_pattern.fullmatch(ss)
				if match is not None:
					if ss == match[0]:
						print('Numbered heading incorrect (matched): {}.'.format(ss))
					else:
						print('Numbered heading incorrect (partially matched): {} != {}.'.format(match[0], ss))

		#--------------------
		# For strict heading.
		strict_heading_pattern = re.compile(r'(\(?([A-Za-z]\d*\.)?(\d+\.)*\d+\)|[A-Za-z]\d*\.(\d+\.)*|(\d+\.)+|\(?[A-Za-z]\d*\)|[A-Za-z]\d*\.)\s+.+')

		if False:
			for head_fmt in numbered_heading_formats:
				for numbering in numberings_correct:
					ss = head_fmt.format(numbering)

					match = strict_heading_pattern.match(ss)
					if match is None:
						print('Numbered heading correct (unmatched): {}.'.format(ss))
					elif ss != match[0]:
						print('Numbered heading correct (partially matched): {} != {}.'.format(match[0], ss))

				for numbering in numberings_incorrect:
					ss = head_fmt.format(numbering)

					#match = strict_heading_pattern.match(ss)
					match = strict_heading_pattern.fullmatch(ss)
					if match is not None:
						if ss == match[0]:
							print('Numbered heading incorrect (matched): {}.'.format(ss))
						else:
							print('Numbered heading incorrect (partially matched): {} != {}.'.format(match[0], ss))

		headings_correct = [
		]
		headings_incorrect = [
			'18', '128.', '235)', '(921)'
			'5.8 (3.5 - 8.2)', '7.8 (5.2–14) respectively.', '0.7 mm', '3.4 >10 min',
			'97.6 ± 1.38', '2.2 to 20.3', '8.61 ± 0.31 ng/ml, of obstructive hydrocephalus was',
			'E. C. Cashman1 and M. J. Donnelly2', 'M. E. Huth,1, 2 A. J. Ricci,1, 3 and A. G. Cheng1', 'E. C. Cashman,1 Terence Farrell,2 and M. Shandilya1', 'J. Law*, P. Shaw , K. Earland , M. Sheldon and M. Lee',
			'A. Maria et al.',
			'P. putida was inoculated with B',
			'P. putida was inoculated with B.',  # NOTE [info] >> re.fullmatch() is stuck. This is a bug in re library.
		]

		# REF [function] >> misc_expressions()
		paper_author_list_pattern = re.compile(r'(([Aa][Nn][Dd]\s+)?(\w+\s+)*([A-Z]\.\s*)*((?![Aa][Nn][Dd]\s+)[-\w]+\s*)+(,(\s*\*)?\s*|\*(\s*,)?\s*|,\s*(\d+|\*)\s*|(\d+|\*)(\s*,)?\s*)*)+')

		for ss in headings_correct:
			match = strict_heading_pattern.match(ss)
			if match is None:
				print('Heading correct (unmatched): {}.'.format(ss))
			elif ss != match[0]:
				print('Heading correct (partially matched): {} != {}.'.format(match[0], ss))
			else:
				match_author = paper_author_list_pattern.fullmatch(ss)
				#if match_author is not None and ss == match_author[0]:
				if match_author is not None:
					print('Heading correct (paper author list): {}.'.format(ss))

		for ss in headings_incorrect:
			#match = strict_heading_pattern.match(ss)
			match = strict_heading_pattern.fullmatch(ss)
			if match is not None:
				#match_author = paper_author_list_pattern.fullmatch(ss)
				#if match_author is None:
				match_author = paper_author_list_pattern.match(ss)
				if match_author is None or ss != match_author[0]:
					if ss == match[0]:
						# NOTE [info] >> It's a trick.
						if ss.lower().find('et al.') < 0:
							print('Heading incorrect (matched): {}.'.format(ss))
					elif ss != match[0]:
						print('Heading incorrect (partially matched): {} != {}.'.format(match[0], ss))
	
		#--------------------
		# For special heading.
		re_pattern = re.compile(r'([Tt][Aa][Bb][Ll][Ee]|[Ff][Ii][Gg][Uu][Rr][Ee]|[Aa][Pp]{2}[Ee][Nn][Dd][Ii][Xx])\s+\d+\.?')

		special_headings_correct = [
			'Table 0', 'TABLE 1.',
			'figure 13', 'FIGURE 37.',
			'APPendix 297', 'AppEndiX 987.',
		]
		special_headings_incorrect = [
			'Tablea 0', 'TABLE 1)',
			'figure (13', 'PIGURE 37.',
			'APPendix 297]', '0AppEndiX 987.',
		]

		for ss in special_headings_correct:
			match = re_pattern.match(ss)
			if match is None:
				print('Special heading correct (unmatched): {}.'.format(ss))
			elif ss != match[0]:
				print('Special heading correct (partially matched): {} != {}.'.format(match[0], ss))

		for ss in special_headings_incorrect:
			#match = re_pattern.match(ss)
			match = re_pattern.fullmatch(ss)
			if match is not None:
				if ss == match[0]:
					print('Special heading incorrect (matched): {}.'.format(ss))
				elif ss != match[0]:
					print('Special heading incorrect (partially matched): {} != {}.'.format(match[0], ss))

	#--------------------
	# For list.
	if False:
		#re_pattern = re.compile(r'(\(?\d+\)|\d+\.|\(?[a-zA-Z]\)|[a-zA-Z]\.)')  # Not correct.
		#re_pattern = re.compile(r'\(?\b\d+\b\)|\b\d+\b\.|\(?\b[a-zA-Z]\b\)|\b[a-zA-Z]\b\.')  # Digit and alphabet.
		#re_pattern = re.compile(r'(\(?\b\d+\b\)|\b\d+\b\.|\(?\b[a-zA-Z]\b\)|\b[a-zA-Z]\b\.)')  # Digit and alphabet.
		#re_pattern = re.compile(r'\(?\b(?:i{1,3}|i[vx]|[vx]i{0,3})\b\)|\b(?:i{1,3}|i[vx]|[vx]i{0,3})\b\.|\(?\b(?:I{1,3}|I[VX]|[VX]I{0,3})\b\)|\b(?:I{1,3}|I[VX]|[VX]I{0,3})\b\.')  # Roman numerals (1 ~ 13).
		#re_pattern = re.compile(r'(\(?\b(?:i{1,3}|i[vx]|[vx]i{0,3})\b\)|\b(?:i{1,3}|i[vx]|[vx]i{0,3})\b\.|\(?\b(?:I{1,3}|I[VX]|[VX]I{0,3})\b\)|\b(?:I{1,3}|I[VX]|[VX]I{0,3})\b\.)')  # Roman numerals (1 ~ 13).
		#re_pattern = re.compile(r'\(?\b\d+\b\)|\b\d+\b\.|\(?\b[a-zA-Z]\b\)|\b[a-zA-Z]\b\.|\(?\b(?:i{1,3}|i[vx]|[vx]i{0,3})\b\)|\b(?:i{1,3}|i[vx]|[vx]i{0,3})\b\.|\(?\b(?:I{1,3}|I[VX]|[VX]I{0,3})\b\)|\b(?:I{1,3}|I[VX]|[VX]I{0,3})\b\.|•')
		re_pattern = re.compile(r'(\(?\b\d+\b\)|\b\d+\b\.|\(?\b[a-zA-Z]\b\)|\b[a-zA-Z]\b\.|\(?\b(?:i{1,3}|i[vx]|[vx]i{0,3})\b\)|\b(?:i{1,3}|i[vx]|[vx]i{0,3})\b\.|\(?\b(?:I{1,3}|I[VX]|[VX]I{0,3})\b\)|\b(?:I{1,3}|I[VX]|[VX]I{0,3})\b\.|•)')

		list_paragraph_formats = [
			'{1} studies not conducted in SSA{0} {2} studies that were reviews, case reports, or case series{0} {3} studies that did not relate epidemiologic data{0} {4} studies that did not discuss dementia or its subtypes in individuals aged 60 years and above.',
			#'{1} studies not conducted in SSA{0} {2} studies that were reviews, case reports, or case series{0} {3} studies that did not relate epidemiologic data{0} {4} studies that did not discuss dementia or its subtypes in individuals aged 60 years and above. VII.',
			#'studies not conducted in SSA{0} studies that were reviews, case reports, or case series{0} studies that did not relate epidemiologic data{0} studies that did not discuss dementia or its subtypes in individuals aged 60 years and above.',
			'{1} WG: weight gain{0} {2} EWG: excessive weight gain{0} {3} kg: kilograms{0} {4} OB: obese; BMI: body mass index; MH: ministry of health.',
			'{1}. {2}, {3}; {4}:',
			#'{1} {2}',
		]
		list_separators = [';', ':', ',', '.']

		list_numbering_formats = ['({})', '{})', '{}.']
		list_indices_lst = [
			[1, 2, 3, 4], [3, 4, 5, 6], [9, 10, 11, 12], [1279, 1280, 1281, 1282],
			['a', 'b', 'c', 'd'], ['c', 'd', 'e', 'f'],
			['A', 'B', 'C', 'D'], ['C', 'D', 'E', 'F'],
			['i', 'ii', 'iii', 'iv'], ['iii', 'iv', 'v', 'vi'], ['ix', 'x', 'xi', 'xii'],
			['I', 'II', 'III', 'IV'], ['III', 'IV', 'V', 'VI'], ['IX', 'X', 'XI', 'XII'],
			# Error cases.
			#['XIi', 'iXII', '0XII', 'XI0'],
		]
		list_bullets = ['•', '‣']

		for para_fmt in list_paragraph_formats:
			for lst_sep in list_separators:
				for num_fmt in list_numbering_formats:
					for indices in list_indices_lst:
						num_indices = [num_fmt.format(idx) for idx in indices]
						ss = para_fmt.format(lst_sep, *num_indices)

						matches_found = re_pattern.findall(ss)
						if matches_found != num_indices:
							print('Matches found (len = {}): {}.'.format(len(matches_found), matches_found))
							occurrences = re_pattern.split(ss)
							print('Occurrences split (len = {}): {}.'.format(len(occurrences), occurrences))

				for bullet in list_bullets:
					bullets = [bullet] * 4
					ss = para_fmt.format(lst_sep, *bullets)

					matches_found = re_pattern.findall(ss)
					if matches_found != bullets:
						print('Matches found (len = {}): {}.'.format(len(matches_found), matches_found))
						occurrences = re_pattern.split(ss)
						print('Occurrences split (len = {}): {}.'.format(len(occurrences), occurrences))

def main():
	#basic_operation()
	#pattern_object_example()
	#match_object_operation()

	#--------------------
	#commonly_used_expressions()
	#misc_expressions()

	page_object_example()  # Numbering, heading, list.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
