#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import enchant
from enchant.checker import SpellChecker
from enchant.tokenize import EmailFilter, URLFilter
from enchant.tokenize import get_tokenizer, HTMLChunker

# REF [site] >> https://pyenchant.github.io/pyenchant/tutorial.html
def tutorial():
	d = enchant.Dict('en_US')
	print("d.check('Hello') =", d.check('Hello'))
	print("d.check('Helo') =", d.check('Helo'))

	print("enchant.dict_exists('fake') =", enchant.dict_exists('fake'))
	print("enchant.dict_exists('en_US') =", enchant.dict_exists('en_US'))
	print('enchant.list_languages() =', enchant.list_languages())

	d = enchant.request_dict('en_US')

	print("d.suggest('Helo') =", d.suggest('Helo'))

	# Personal word lists (PWL).
	d1 = enchant.request_pwl_dict('./mywords.txt')
	d2 = enchant.DictWithPWL('en_US', './mywords.txt')

	print("d2.check('Hello') =", d2.check('Hello'))

	# Check entire blocks of text.
	chkr = SpellChecker('en_US')
	chkr.set_text('This is sme sample txt with erors.')
	for err in chkr:
		print('ERROR:', err.word)

	# The SpellChecker can use filters to ignore certain word forms.
	chkr = SpellChecker('en_US', filters=[EmailFilter, URLFilter])

	# Tokenization: splitting text into words.
	tknzr = get_tokenizer('en_US')
	print('Tokenizer1 result =', [w for w in tknzr('this is some simple text')])

	print('Tokenizer1 result =', [w for w in tknzr("this is <span class='important'>really important</span> text")])

	tknzr2 = get_tokenizer('en_US', chunkers=(HTMLChunker,))
	print('Tokenizer2 result =', [w for w in tknzr2("this is <span class='important'>really important</span> text")])

	print('Tokenizer1 result =', [w for w in tknzr('send an email to fake@example.com please')])

	tknzr3 = get_tokenizer('en_US', [EmailFilter])
	print('Tokenizer3 result =', [w for w in tknzr3('send an email to fake@example.com please')])

def main():
	tutorial()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
