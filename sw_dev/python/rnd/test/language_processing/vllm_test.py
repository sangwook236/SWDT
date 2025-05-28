#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://docs.vllm.ai/en/latest/getting_started/quickstart.html
def quickstart():
	if True:
		# Offline Batched Inference

		from vllm import LLM, SamplingParams

		prompts = [
			"Hello, my name is",
			"The president of the United States is",
			"The capital of France is",
			"The future of AI is",
		]
		sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
		#sampling_params = SamplingParams(temperature=0.0, top_p=1.0)  # Greedy decoding

		llm = LLM(model="facebook/opt-125m")
		#llm = LLM(model="google/flan-t5-xxl")
		#llm = LLM("facebook/opt-13b", tensor_parallel_size=4)

		outputs = llm.generate(prompts, sampling_params)

		for output in outputs:
			prompt = output.prompt
			generated_text = output.outputs[0].text
			print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

	if True:
		# Streaming responses (for longer sequences and real-time feedback)

		from vllm import LLM, SamplingParams

		prompts = [
			"Write a short story about a robot who learns to love."
		]
		sampling_params = SamplingParams(temperature=0.8, max_tokens=512)

		llm = LLM(model="facebook/opt-125m")

		outputs = llm.generate(prompts, sampling_params, stream=True)

		for output in outputs:
			prompt = output.prompt
			print(f"Prompt: {prompt!r}")
			print("Generated text:")
			for chunk in output.outputs[0].chunks:
				print(chunk.text, end="", flush=True)
			print("\n")

	#-----
	# OpenAI-Compatible Server
	#	vLLM can be deployed as a server that implements the OpenAI API protocol.
	#	This allows vLLM to be used as a drop-in replacement for applications using OpenAI API.
	#	By default, it starts the server at http://localhost:8000.
	#	You can specify the address with --host and --port arguments.
	#		vllm serve --model <MODE_NAME> --host 0.0.0.0 --port 8000  # For remote access
	#	Run the following command to start the vLLM server with the Qwen2.5-1.5B-Instruct model:
	#		vllm serve Qwen/Qwen2.5-1.5B-Instruct
	#	This server can be queried in the same format as OpenAI API. For example, to list the models:
	#		curl http://localhost:8000/v1/models
	#	You can pass in the argument --api-key or environment variable VLLM_API_KEY to enable the server to check for API key in the header.

	# Install:
	#	pip install openai

	if True:
		# OpenAI Completions API with vLLM

		# Once your server is started, you can query the model with input prompts:
		#	curl http://localhost:8000/v1/completions \
		#		-H "Content-Type: application/json" \
		#		-d '{
		#			"model": "Qwen/Qwen2.5-1.5B-Instruct",
		#			"prompt": "San Francisco is a",
		#			"max_tokens": 7,
		#			"temperature": 0
		#		}'

		# Use the OpenAI client library
		from openai import OpenAI

		# Modify OpenAI's API key and API base to use vLLM's API server.
		openai_api_key = "EMPTY"
		openai_api_base = "http://localhost:8000/v1"

		client = OpenAI(
			api_key=openai_api_key,
			base_url=openai_api_base,
		)

		if True:
			model = "Qwen/Qwen2.5-1.5B-Instruct"
		else:
			models = client.models.list()
			model = models.data[0].id

		completion = client.completions.create(
			model=model,
			prompt="San Francisco is a",
			max_tokens=7,
			temperature=0
		)
		print("Completion result:", completion)

		# REF [site] >> https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_completion_client.py

	if True:
		# OpenAI Chat Completions API with vLLM

		# You can use the create chat completion endpoint to interact with the model:
		#	curl http://localhost:8000/v1/chat/completions \
		#		-H "Content-Type: application/json" \
		#		-d '{
		#			"model": "Qwen/Qwen2.5-1.5B-Instruct",
		#			"messages": [
		#				{"role": "system", "content": "You are a helpful assistant."},
		#				{"role": "user", "content": "Who won the world series in 2020?"}
		#			]
		#		}'

		from openai import OpenAI

		# Set OpenAI's API key and API base to use vLLM's API server.
		openai_api_key = "EMPTY"
		openai_api_base = "http://localhost:8000/v1"

		client = OpenAI(
			api_key=openai_api_key,
			base_url=openai_api_base,
		)

		if True:
			model = "Qwen/Qwen2.5-1.5B-Instruct"
		else:
			models = client.models.list()
			model = models.data[0].id

		chat_response = client.chat.completions.create(
			model=model,
			messages=[
				{"role": "system", "content": "You are a helpful assistant."},
				{"role": "user", "content": "Tell me a joke."},
			]
		)
		print("Chat response:", chat_response)

def main():
	# Install:
	#	pip install vllm

	quickstart()

	# Examples:
	#	https://docs.vllm.ai/en/latest/getting_started/examples/examples_index.html
	#	https://github.com/vllm-project/vllm/blob/main/docs/source/generate_examples.py
	#	https://github.com/vllm-project/vllm/tree/main/examples
	#		Offline inference
	#		Online serving

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
