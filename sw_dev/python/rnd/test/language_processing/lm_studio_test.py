#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import lmstudio as lms

# REF [site] >> https://lmstudio.ai/docs/developer
def simple_example():
	with lms.Client() as client:
		model = client.llm.model("openai/gpt-oss-20b")
		result = model.respond("Who are you, and what can you do?")
		print(result)

def main():
	# References
	#	https://lmstudio.ai/docs/python

	# Install:
	#	pip install lmstudio

	simple_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
