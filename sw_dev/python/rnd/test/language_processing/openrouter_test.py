#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://openrouter.ai/docs/quickstart
def quickstart():
	if True:
		# API

		import requests
		import json

		response = requests.post(
			url="https://openrouter.ai/api/v1/chat/completions",
			headers={
				"Authorization": "Bearer <OPENROUTER_API_KEY>",
				"HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
				"X-OpenRouter-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
			},
			data=json.dumps({
				"model": "~openai/gpt-latest",
				"messages": [
				{
					"role": "user",
					"content": "What is the meaning of life?"
				}
				]
			})
		)

	if True:
		# Client SDKs

		# Install:
		#	pip install openrouter

		from openrouter import OpenRouter
		import os

		with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
			response = client.chat.send(
				model="~openai/gpt-latest",
				messages=[
					{"role": "user", "content": "What is the meaning of life?"}
				],
			)

			print(response.choices[0].message.content)

	if False:
		# Agent SDK

		# No support for agent SDK yet
		raise NotSupportedError

def main():
	# Install:
	#	pip install openrouter

	quickstart()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
