#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://docs.sglang.io/basic_usage/openai_api.html
def openai_compatiable_apis_example():
	raise NotImplementedError

# REF [site] >> https://docs.sglang.io/basic_usage/ollama_api.html
def ollama_compatiable_apis_example():
	raise NotImplementedError

# REF [site] >> https://docs.sglang.io/basic_usage/offline_engine_api.html
def offline_batch_inference_example():
	# Launch the offline engine
	import asyncio

	import sglang as sgl
	import sglang.test.doc_patch
	from sglang.utils import async_stream_and_merge, stream_and_merge

	llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")

	if True:
		# Non-streaming Synchronous Generation

		prompts = [
			"Hello, my name is",
			"The president of the United States is",
			"The capital of France is",
			"The future of AI is",
		]

		sampling_params = {"temperature": 0.8, "top_p": 0.95}

		outputs = llm.generate(prompts, sampling_params)
		for prompt, output in zip(prompts, outputs):
			print("===============================")
			print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

	if True:
		# Streaming Synchronous Generation

		prompts = [
			"Write a short, neutral self-introduction for a fictional character. Hello, my name is",
			"Provide a concise factual statement about France’s capital city. The capital of France is",
			"Explain possible future trends in artificial intelligence. The future of AI is",
		]

		sampling_params = {
			"temperature": 0.2,
			"top_p": 0.9,
		}

		print("\n=== Testing synchronous streaming generation with overlap removal ===\n")

		for prompt in prompts:
			print(f"Prompt: {prompt}")
			merged_output = stream_and_merge(llm, prompt, sampling_params)
			print("Generated text:", merged_output)
			print()

	if True:
		# Non-streaming Asynchronous Generation

		prompts = [
			"Write a short, neutral self-introduction for a fictional character. Hello, my name is",
			"Provide a concise factual statement about France’s capital city. The capital of France is",
			"Explain possible future trends in artificial intelligence. The future of AI is",
		]

		sampling_params = {"temperature": 0.8, "top_p": 0.95}

		print("\n=== Testing asynchronous batch generation ===")

		async def main():
			outputs = await llm.async_generate(prompts, sampling_params)

			for prompt, output in zip(prompts, outputs):
				print(f"\nPrompt: {prompt}")
				print(f"Generated text: {output['text']}")

		asyncio.run(main())

	if True:
		# Streaming Asynchronous Generation

		prompts = [
			"Write a short, neutral self-introduction for a fictional character. Hello, my name is",
			"Provide a concise factual statement about France’s capital city. The capital of France is",
			"Explain possible future trends in artificial intelligence. The future of AI is",
		]

		sampling_params = {"temperature": 0.8, "top_p": 0.95}

		print("\n=== Testing asynchronous streaming generation (no repeats) ===")

		async def main():
			for prompt in prompts:
				print(f"\nPrompt: {prompt}")
				print("Generated text: ", end="", flush=True)

				# Replace direct calls to async_generate with our custom overlap-aware version
				async for cleaned_chunk in async_stream_and_merge(llm, prompt, sampling_params):
					print(cleaned_chunk, end="", flush=True)

				print()  # New line after each prompt

		asyncio.run(main())

def main():
	# Install:
	#	pip install --upgrade pip

	#openai_compatiable_apis_example()  # Not yet implemented
	#ollama_compatiable_apis_example()  # Not yet implemented

	offline_batch_inference_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
