#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://dspy.ai/
def get_started_1():
	if True:
		import dspy

		if True:
			lm = dspy.LM("openai/gpt-5-mini", api_key="YOUR_OPENAI_API_KEY")
		elif False:
			lm = dspy.LM("anthropic/claude-sonnet-4-5-20250929", api_key="YOUR_ANTHROPIC_API_KEY")
		elif False:
			lm = dspy.LM(
				"databricks/databricks-llama-4-maverick",
				api_key="YOUR_DATABRICKS_ACCESS_TOKEN",
				api_base="YOUR_DATABRICKS_WORKSPACE_URL",  # e.g.: https://dbc-64bf4923-e39e.cloud.databricks.com/serving-endpoints
			)
		elif False:
			lm = dspy.LM("gemini/gemini-2.5-flash", api_key="YOUR_GEMINI_API_KEY")
		elif False:
			# Local LMs on your laptop

			# Install:
			#	curl -fsSL https://ollama.ai/install.sh | sh
			# Run:
			#	ollama run llama3.2:1b

			lm = dspy.LM("ollama_chat/llama3.2:1b", api_base="http://localhost:11434", api_key="")
		elif False:
			# Local LMs on a GPU server

			# Install:
			#	pip install "sglang[all]"
			#	pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ 
			# Run:
			#	CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path meta-llama/Llama-3.1-8B-Instruct

			lm = dspy.LM(
				"openai/meta-llama/Llama-3.1-8B-Instruct",
				api_base="http://localhost:7501/v1",  # ensure this points to your port
				api_key="local", model_type="chat"
			)
		elif False:
			# Other providers

			lm = dspy.LM("openai/your-model-name", api_key="PROVIDER_API_KEY", api_base="YOUR_PROVIDER_URL")
		dspy.configure(lm=lm)

		lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']
		lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']

# REF [site] >> https://dspy.ai/
def get_started_2():
	import dspy

	if True:
		# Math

		math = dspy.ChainOfThought("question -> answer: float")
		math(question="Two dice are tossed. What is the probability that the sum equals two?")

	if True:
		# RAG

		def search_wikipedia(query: str) -> list[str]:
			results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
			return [x["text"] for x in results]

		rag = dspy.ChainOfThought("context, question -> response")

		question = "What's the name of the castle that David Gregory inherited?"
		rag(context=search_wikipedia(question), question=question)

	if True:
		# Classification

		from typing import Literal

		class Classify(dspy.Signature):
			"""Classify sentiment of a given sentence."""

			sentence: str = dspy.InputField()
			sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
			confidence: float = dspy.OutputField()

		classify = dspy.Predict(Classify)
		classify(sentence="This book was super fun to read, though not the last chapter.")

	if True:
		# Information extraction

		class ExtractInfo(dspy.Signature):
			"""Extract structured information from text."""

			text: str = dspy.InputField()
			title: str = dspy.OutputField()
			headings: list[str] = dspy.OutputField()
			entities: list[dict[str, str]] = dspy.OutputField(desc="a list of entities and their metadata")

		module = dspy.Predict(ExtractInfo)

		text = "Apple Inc. announced its latest iPhone 14 today." \
			"The CEO, Tim Cook, highlighted its new features in a press release."
		response = module(text=text)

		print(response.title)
		print(response.headings)
		print(response.entities)

	if True:
		# Agents

		def evaluate_math(expression: str):
			return dspy.PythonInterpreter({}).execute(expression)

		def search_wikipedia(query: str):
			results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
			return [x["text"] for x in results]

		react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

		pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
		print(pred.answer)

	if True:
		# Multi-stage pipelines

		class Outline(dspy.Signature):
			"""Outline a thorough overview of a topic."""

			topic: str = dspy.InputField()
			title: str = dspy.OutputField()
			sections: list[str] = dspy.OutputField()
			section_subheadings: dict[str, list[str]] = dspy.OutputField(desc="mapping from section headings to subheadings")

		class DraftSection(dspy.Signature):
			"""Draft a top-level section of an article."""

			topic: str = dspy.InputField()
			section_heading: str = dspy.InputField()
			section_subheadings: list[str] = dspy.InputField()
			content: str = dspy.OutputField(desc="markdown-formatted section")

		class DraftArticle(dspy.Module):
			def __init__(self):
				self.build_outline = dspy.ChainOfThought(Outline)
				self.draft_section = dspy.ChainOfThought(DraftSection)

			def forward(self, topic):
				outline = self.build_outline(topic=topic)
				sections = []
				for heading, subheadings in outline.section_subheadings.items():
					section, subheadings = f"## {heading}", [f"### {subheading}" for subheading in subheadings]
					section = self.draft_section(topic=outline.title, section_heading=section, section_subheadings=subheadings)
					sections.append(section.content)
				return dspy.Prediction(title=outline.title, sections=sections)

		draft_article = DraftArticle()
		article = draft_article(topic="World Cup 2002")

def main():
	# Install:
	#	pip install -U dspy

	get_started_1()
	get_started_2()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
