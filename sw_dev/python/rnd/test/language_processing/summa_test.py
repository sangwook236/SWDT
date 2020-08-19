#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://pypi.org/project/summa/

import summa

def simple_example():
	text = """Automatic summarization is the process of reducing a text document with a \
computer program in order to create a summary that retains the most important points \
of the original document. As the problem of information overload has grown, and as \
the quantity of data has increased, so has interest in automatic summarization. \
Technologies that can make a coherent summary take into account variables such as \
length, writing style and syntax. An example of the use of summarization technology \
is search engines such as Google. Document summarization is another."""

	print('summa.summarizer.summarize(text) =\n', summa.summarizer.summarize(text))
	print('summa.summarizer.summarize(text, ratio=0.2) =\n', summa.summarizer.summarize(text, ratio=0.2))
	print('summa.summarizer.summarize(text, words=50) =\n', summa.summarizer.summarize(text, words=50))
	print('summa.summarizer.summarize(text, split=True) =\n', summa.summarizer.summarize(text, split=True))

	# The available languages: Arabic, Danish, Dutch, English, Finnish, French, German, Hungarian, Italian, Norwegian, Polish, Porter, Portuguese, Romanian, Russian, Spanish, Swedish.
	print("summa.summarizer.summarize(text, language='spanish') =\n", summa.summarizer.summarize(text, language='spanish'))

	print('summa.keywords.keywords(text) =\n', summa.keywords.keywords(text))

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
