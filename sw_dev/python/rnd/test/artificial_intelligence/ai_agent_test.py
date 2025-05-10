#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://huggingface.co/docs/smolagents/guided_tour
def smolagents_guided_tour():
	# Install:
	#	pip install smolagents

	from smolagents import CodeAgent

	#-----
	# Build your agent

	# To initialize a minimal agent, you need at least these two arguments:
	#	model, a text-generation model to power your agent - because the agent is different from a simple LLM, it is a system that uses a LLM as its engine. You can use any of these options:
	#		TransformersModel takes a pre-initialized transformers pipeline to run inference on your local machine using transformers.
	#		InferenceClientModel leverages a huggingface_hub.InferenceClient under the hood and supports all Inference Providers on the Hub: Cerebras, Cohere, Fal, Fireworks, HF-Inference, Hyperbolic, Nebius, Novita, Replicate, SambaNova, Together, and more.
	#		LiteLLMModel similarly lets you call 100+ different models and providers through LiteLLM!
	#		AzureOpenAIServerModel allows you to use OpenAI models deployed in Azure.
	#		AmazonBedrockServerModel allows you to use Amazon Bedrock in AWS.
	#		MLXModel creates a mlx-lm pipeline to run inference on your local machine.
	#	tools, a list of Tools that the agent can use to solve the task. It can be an empty list. You can also add the default toolbox on top of your tools list by defining the optional argument add_base_tools=True.

	if True:
		# Install:
		#	pip install smolagents[transformers]

		from smolagents import TransformersModel

		model_id = "meta-llama/Llama-3.2-3B-Instruct"

		model = TransformersModel(model_id=model_id)
		#agent = CodeAgent(tools=[], model=model, add_base_tools=True)
	elif False:
		# Inference Providers need a HF_TOKEN to authenticate, but a free HF account already comes with included credits

		from smolagents import InferenceClientModel

		model_id = "meta-llama/Llama-3.3-70B-Instruct" 

		model = InferenceClientModel(model_id=model_id, token="<YOUR_HUGGINGFACEHUB_API_TOKEN>")  # You can choose to not pass any model_id to InferenceClientModel to use a default model
		# You can also specify a particular provider e.g. provider="together" or provider="sambanova"
		#agent = CodeAgent(tools=[], model=model, add_base_tools=True)
	elif False:
		# To use LiteLLMModel, you need to set the environment variable ANTHROPIC_API_KEY or OPENAI_API_KEY, or pass api_key variable upon initialization.
	
		# Install:
		#	pip install smolagents[litellm]

		from smolagents import LiteLLMModel

		model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", api_key="YOUR_ANTHROPIC_API_KEY")
		#model = LiteLLMModel(model_id="openai/gpt-4o", api_key="OPENAI_API_KEY")
		#agent = CodeAgent(tools=[], model=model, add_base_tools=True)
	elif False:
		# Install:
		#	pip install smolagents[litellm]

		from smolagents import LiteLLMModel

		model = LiteLLMModel(
			model_id="ollama_chat/llama3.2",  # This model is a bit weak for agentic behaviours though
			api_base="http://localhost:11434",  # Replace with 127.0.0.1:11434 or remote open-ai compatible server if necessary
			api_key="YOUR_API_KEY",  # Replace with API key if necessary
			num_ctx=8192,  # ollama default is 2048 which will fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
		)
		#agent = CodeAgent(tools=[], model=model, add_base_tools=True)

	#-----
	# CodeAgent and ToolCallingAgent

	# The CodeAgent is our default agent. It will write and execute python code snippets at each step.
	# By default, the execution is done in your local environment. This should be safe because the only functions that can be called are the tools you provided (especially if it's only tools by Hugging Face) and a set of predefined safe functions like print or functions from the math module, so you're already limited in what can be executed.
	# The Python interpreter also doesn't allow imports by default outside of a safe list, so all the most obvious attacks shouldn't be an issue. You can authorize additional imports by passing the authorized modules as a list of strings in argument additional_authorized_imports upon initialization of your CodeAgent:
	# We also support the widely-used way of writing actions as JSON-like blobs: this is ToolCallingAgent, it works much in the same way like CodeAgent, of course without additional_authorized_imports since it doesn,t execute code:

	if True:
		agent = CodeAgent(tools=[], model=model, add_base_tools=True)
		agent.run("Could you give me the 118th number in the Fibonacci sequence?",)
	elif False:
		agent = CodeAgent(tools=[], model=model, additional_authorized_imports=["requests", "bs4"])
		agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
	elif False:
		from smolagents import ToolCallingAgent

		agent = ToolCallingAgent(tools=[], model=model)
		agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")

	#-----
	# Inspect an agent run

	# Here are a few useful attributes to inspect what happened after a run:
	#	agent.logs stores the fine-grained logs of the agent. At every step of the agent's run, everything gets stored in a dictionary that then is appended to agent.logs.
	#	Running agent.write_memory_to_messages() writes the agent's memory as list of chat messages for the Model to view.
	#	This method goes over each step of the log and only stores what it's interested in as a message: for instance, it will save the system prompt and task in separate messages, then for each step it will store the LLM output as a message, and the tool call output as another message.
	#	Use this if you want a higher-level view of what has happened - but not every log will be transcripted by this method.

	#-----
	# Tools

	# A tool is an atomic function to be used by an agent. To be used by an LLM, it also needs a few attributes that constitute its API and will be used to describe to the LLM how to call this tool:
	#	- A name
	#	- A description
	#	- Input types and descriptions
	#	- An output type
	# You can for instance check the PythonInterpreterTool: it has a name, a description, input descriptions, an output type, and a forward method to perform the action.
	# When the agent is initialized, the tool attributes are used to generate a tool description which is baked into the agent's system prompt. This lets the agent know which tools it can use and why.

	# Default toolbox

	# If you install smolagents with the "toolkit" extra, it comes with a default toolbox for empowering agents, that you can add to your agent upon initialization with argument add_base_tools=True:
	#	- DuckDuckGo web search*: performs a web search using DuckDuckGo browser.
	#	- Python code interpreter: runs your LLM generated Python code in a secure environment. This tool will only be added to ToolCallingAgent if you initialize it with add_base_tools=True, since code-based agent can already natively execute Python code
	#	- Transcriber: a speech-to-text pipeline built on Whisper-Turbo that transcribes an audio to text.

	if False:
		# You can manually use a tool by calling it with its arguments.

		# Install:
		#	pip install smolagents[toolkit]

		from smolagents import WebSearchTool

		search_tool = WebSearchTool()
		print(search_tool("Who's the current president of Russia?"))

	# Create a new tool
	#	You can create your own tool for use cases not covered by the default tools from Hugging Face.

	from huggingface_hub import list_models

	if False:
		# For example, let's create a tool that returns the most downloaded model for a given task from the Hub.

		task = "text-classification"

		most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
		print(most_downloaded_model.id)

	# This code can quickly be converted into a tool, just by wrapping it in a function and adding the tool decorator.
	# This is not the only way to build the tool: you can directly define it as a subclass of Tool, which gives you more flexibility, for instance the possibility to initialize heavy class attributes.
	if True:
		from smolagents import tool

		@tool
		def model_download_tool(task: str) -> str:
			"""
			This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
			It returns the name of the checkpoint.

			Args:
				task: The task for which to get the download count.
			"""
			most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
			return most_downloaded_model.id

		# The function needs:
		#	- A clear name. The name should be descriptive enough of what this tool does to help the LLM brain powering the agent. Since this tool returns the model with the most downloads for a task, let's name it model_download_tool.
		#	- Type hints on both inputs and output
		#	- A description, that includes an 'Args:' part where each argument is described (without a type indication this time, it will be pulled from the type hint). Same as for the tool name, this description is an instruction manual for the LLM powering your agent, so do not neglect it.
	else:
		from smolagents import Tool

		class ModelDownloadTool(Tool):
			name = "model_download_tool"
			description = "This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub. It returns the name of the checkpoint."
			inputs = {"task": {"type": "string", "description": "The task for which to get the download count."}}
			output_type = "string"

			def forward(self, task: str) -> str:
				most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
				return most_downloaded_model.id

		# The subclass needs the following attributes:
		#	- A clear name. The name should be descriptive enough of what this tool does to help the LLM brain powering the agent. Since this tool returns the model with the most downloads for a task, let's name it model_download_tool.
		#	- A description. Same as for the name, this description is an instruction manual for the LLM powering your agent, so do not neglect it.
		#	- Input types and descriptions
		#	- Output type All these attributes will be automatically baked into the agent's system prompt upon initialization: so strive to make them as clear as possible!

	# Initialize your agent

	from smolagents import InferenceClientModel

	agent = CodeAgent(tools=[model_download_tool], model=InferenceClientModel())
	agent.run(
		"Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
	)

	#-----
	# Multi-agents

	from smolagents import CodeAgent, InferenceClientModel, WebSearchTool

	model = InferenceClientModel()

	web_agent = CodeAgent(
		tools=[WebSearchTool()],
		model=model,
		name="web_search",
		description="Runs web searches for you. Give it your query as an argument."
	)

	manager_agent = CodeAgent(
		tools=[], model=model, managed_agents=[web_agent]
	)

	manager_agent.run("Who is the CEO of Hugging Face?")

	# Talk with your agent and visualize its thoughts in a cool Gradio interface

	from smolagents import load_tool, CodeAgent, InferenceClientModel, GradioUI

	# Import tool from Hub
	image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

	model = InferenceClientModel(model_id=model_id)

	# Initialize the agent with the image generation tool
	agent = CodeAgent(tools=[image_generation_tool], model=model)

	GradioUI(agent).launch()

	# Under the hood, when the user types a new answer, the agent is launched with agent.run(user_request, reset=False). The reset=False flag means the agent's memory is not flushed before launching this new task, which lets the conversation go on.
	# You can also use this reset=False argument to keep the conversation going in any other agentic application.
	# In gradio UIs, if you want to allow users to interrupt a running agent, you could do this with a button that triggers method agent.interrupt(). This will stop the agent at the end of its current step, then raise an error.

# REF [site] >> https://huggingface.co/docs/smolagents/tutorials/building_good_agents
def smolagents_building_good_agents_tutorial():
	raise NotImplementedError

# REF [site] >> https://github.com/QwenLM/Qwen-Agent
def qwen_agent_example():
	# Install:
	#	pip install -U "qwen-agent[gui,rag,code_interpreter,mcp]"

	import urllib.parse
	import json5
	from qwen_agent.agents import Assistant
	from qwen_agent.tools.base import BaseTool, register_tool
	from qwen_agent.utils.output_beautify import typewriter_print

	# Step 1 (Optional): Add a custom tool named `my_image_gen`.
	@register_tool("my_image_gen")
	class MyImageGen(BaseTool):
		# The `description` tells the agent the functionality of this tool.
		description = "AI painting (image generation) service, input text description, and return the image URL drawn based on text information."
		# The `parameters` tell the agent what input parameters the tool has.
		parameters = [{
			"name": "prompt",
			"type": "string",
			"description": "Detailed description of the desired image content, in English",
			"required": True
		}]

		def call(self, params: str, **kwargs) -> str:
			# `params` are the arguments generated by the LLM agent.
			prompt = json5.loads(params)["prompt"]
			prompt = urllib.parse.quote(prompt)
			return json5.dumps(
				{"image_url": f"https://image.pollinations.ai/prompt/{prompt}"},
				ensure_ascii=False
			)

	# Step 2: Configure the LLM you are using.
	llm_cfg = {
		# Use the model service provided by DashScope:
		"model": "qwen-max-latest",
		"model_server": "dashscope",
		# "api_key": "YOUR_DASHSCOPE_API_KEY",
		# It will use the `DASHSCOPE_API_KEY` environment variable if 'api_key' is not set here.

		# Use a model service compatible with the OpenAI API, such as vLLM or Ollama:
		# "model": "Qwen2.5-7B-Instruct",
		# "model_server": "http://localhost:8000/v1",  # base_url, also known as api_base
		# "api_key": "EMPTY",

		# (Optional) LLM hyperparameters for generation:
		"generate_cfg": {
			"top_p": 0.8
		}
	}

	# Step 3: Create an agent. Here we use the `Assistant` agent as an example, which is capable of using tools and reading files.
	system_instruction = """After receiving the user's request, you should:
- first draw an image and obtain the image url,
- then run code `request.get(image_url)` to download the image,
- and finally select an image operation from the given document to process the image.
Please show the image using `plt.show()`."""
	tools = ["my_image_gen", "code_interpreter"]  # `code_interpreter` is a built-in tool for executing code.
	files = ["./examples/resource/doc.pdf"]  # Give the bot a PDF file to read.
	bot = Assistant(
		llm=llm_cfg,
		system_message=system_instruction,
		function_list=tools,
		files=files
	)

	# Step 4: Run the agent as a chatbot.
	messages = []  # This stores the chat history.
	while True:
		# For example, enter the query "draw a dog and rotate it 90 degrees".
		query = input("\nuser query: ")
		# Append the user query to the chat history.
		messages.append({"role": "user", "content": query})
		response = []
		response_plain_text = ""
		print("bot response:")
		for response in bot.run(messages=messages):
			# Streaming output.
			response_plain_text = typewriter_print(response, response_plain_text)
		# Append the bot responses to the chat history.
		messages.extend(response)

def main():
	# Protocol:
	#	Model Context Protocol (MCP)
	#	Agent2Agent (A2A) Protocol

	# AutoGen
	#	Refer to autogen_test.py

	# LangGraph
	#	Refer to langchain_test.py

	# NVIDIA Agent Intelligence Toolkit (AIQ Toolkit)
	#	https://developer.nvidia.com/agent-intelligence-toolkit
	#	https://docs.nvidia.com/aiqtoolkit/latest/index.html
	#	https://github.com/NVIDIA/AIQToolkit

	# Smolagents AI Agent Framework
	#	https://smolagents.org/
	#	https://huggingface.co/docs/smolagents/index
	#	https://github.com/huggingface/smolagents
	smolagents_guided_tour()
	#smolagents_building_good_agents_tutorial()  # Not yet implemented

	# Refer to qwen3_example() in hugging_face_transformers_test.py
	qwen_agent_example()  # Qwen-Agent

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
