#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/ollama/ollama-python
def usage():
	if True:
		from ollama import chat
		from ollama import ChatResponse

		response: ChatResponse = chat(
			model="llama3.2",
			messages=[{
				"role": "user",
				"content": "Why is the sky blue?",
			},]
		)
		print(response["message"]["content"])
		# or access fields directly from the response object
		print(response.message.content)

	if True:
		# Streaming responses
		#	Response streaming can be enabled by setting stream=True.

		from ollama import chat

		stream = chat(
			model="llama3.2",
			messages=[{"role": "user", "content": "Why is the sky blue?"}],
			stream=True,
		)

		for chunk in stream:
			print(chunk["message"]["content"], end="", flush=True)

	if True:
		# Custom client
		#	A custom client can be created by instantiating Client or AsyncClient from ollama.
		#	All extra keyword arguments are passed into the httpx.Client.

		from ollama import Client

		client = Client(
			host="http://localhost:11434",
			headers={"x-some-header": "some-value"}
		)
		response = client.chat(
			model="llama3.2",
			messages=[{
				"role": "user",
				"content": "Why is the sky blue?",
			},]
		)

	if True:
		# Async client
		#	The AsyncClient class is used to make asynchronous requests. It can be configured with the same fields as the Client class.

		import asyncio
		from ollama import AsyncClient

		async def chat():
			message = {"role": "user", "content": "Why is the sky blue?"}
			response = await AsyncClient().chat(model="llama3.2", messages=[message])

		asyncio.run(chat())

	if True:
		# Async client
		#	Setting stream=True modifies functions to return a Python asynchronous generator.

		import asyncio
		from ollama import AsyncClient

		async def chat():
			message = {"role": "user", "content": "Why is the sky blue?"}
			async for part in await AsyncClient().chat(model="llama3.2", messages=[message], stream=True):
				print(part["message"]["content"], end="", flush=True)

		asyncio.run(chat())

	if True:
		# API
		#	The Ollama Python library's API is designed around the Ollama REST API.

		import ollama

		# Chat
		ollama.chat(model="llama3.2", messages=[{"role": "user", "content": "Why is the sky blue?"}])

		# Generate
		ollama.generate(model="llama3.2", prompt="Why is the sky blue?")

		# List
		ollama.list()

		# Show
		ollama.show("llama3.2")

		# Create
		ollama.create(model="example", from_="llama3.2", system="You are Mario from Super Mario Bros.")

		# Copy
		ollama.copy("llama3.2", "user/llama3.2")

		# Delete
		ollama.delete("llama3.2")

		# Pull
		ollama.pull("llama3.2")

		# Push
		ollama.push("user/llama3.2")

		# Embed
		ollama.embed(model="llama3.2", input="The sky is blue because of rayleigh scattering")

		# Embed (batch)
		ollama.embed(model="llama3.2", input=["The sky is blue because of rayleigh scattering", "Grass is green because of chlorophyll"])

		# Ps
		ollama.ps()

	if True:
		# Errors
		#	Errors are raised if requests return an error status or if an error is detected while streaming.

		import ollama

		model = "does-not-yet-exist"

		try:
			ollama.chat(model)
		except ollama.ResponseError as ex:
			print("Error:", ex.error)
			if ex.status_code == 404:
				ollama.pull(model)

# REF [site] >> https://github.com/ollama/ollama-python/tree/main/examples
def examples():
	raise NotImplementedError

def main():
	# References
	#	https://ollama.com/
	#	https://ollama.com/search
	#
	#	https://github.com/ollama/ollama

	# Install Ollama:
	#	On Linux:
	#		curl -sSL https://ollama.com/download.sh | sh
	# Run a model:
	#	ollama pull llama3.2
	#	ollama list
	#	ollama run llama3.2
	#	/bye
	#	ollama ps
	#	ollama stop llama3.2
	#	ollama rm llama3.2

	# Install:
	#	pip install ollama

	usage()
	#examples()  # Not yet implemented

	# Web API
	#	qwen_ocr_web_api_test() & llama_ocr_example() in ./ocr_test.py
	# Python API
	#	qwen_vl_ocr_python_test() & llama_ocr_example() in ./ocr_test.py

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
