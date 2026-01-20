#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/BerriAI/litellm
def simple_example():
	if True:
		# Python SDK
		#	https://github.com/BerriAI/litellm

		# Install:
		#	pip install litellm

		import os
		from litellm import completion

		os.environ["OPENAI_API_KEY"] = "your-openai-key"
		os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

		# OpenAI
		response = completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hello!"}])

		# Anthropic  
		response = completion(model="anthropic/claude-sonnet-4-20250514", messages=[{"role": "user", "content": "Hello!"}])

	if True:
		# AI Gateway (Proxy Server)
		#	https://github.com/BerriAI/litellm
		#	https://docs.litellm.ai/docs/proxy/docker_quick_start
		#	https://docs.litellm.ai/docs/providers

		# Install:
		#	pip install 'litellm[proxy]'
		# Run:
		#	litellm --model gpt-4o

		import openai

		client = openai.OpenAI(api_key="anything", base_url="http://0.0.0.0:4000")
		response = client.chat.completions.create(
			model="gpt-4o",
			messages=[{"role": "user", "content": "Hello!"}]
		)

def main():
	# Install:
	#	pip install litellm
	#	pip install 'litellm[proxy]'

	simple_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
