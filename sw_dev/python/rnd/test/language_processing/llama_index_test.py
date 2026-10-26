#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/run-llama/llama_index
def llama_index_example():
	# Install:
	#	pip install llama-index-core
	#	pip install llama-index-llms-openai
	#	pip install llama-index-llms-ollama
	#	pip install llama-index-embeddings-huggingface

	if True:
		# To build a simple vector store index using OpenAI:

		import os
		os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

		from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

		documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
		index = VectorStoreIndex.from_documents(documents)

	if True:
		# To build a simple vector store index using non-OpenAI LLMs, e.g. LLMs hosted through Ollama:

		from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
		from llama_index.embeddings.huggingface import HuggingFaceEmbedding
		from llama_index.llms.ollama import Ollama
		from transformers import AutoTokenizer

		# Set the LLM
		Settings.llm = Ollama(
			model="llama-3.1:latest",
			request_timeout=360.0,
		)

		# Set the tokenizer to match the LLM
		Settings.tokenizer = AutoTokenizer.from_pretrained(
			"meta-llama/Llama-3.1-8B-Instruct"
		)

		# Set the embed model
		Settings.embed_model = HuggingFaceEmbedding(
			model_name="BAAI/bge-small-en-v1.5"
		)

		documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
		index = VectorStoreIndex.from_documents(
			documents,
		)

		# To query:
		query_engine = index.as_query_engine()
		query_engine.query("YOUR_QUESTION")

		# By default, data is stored in-memory. To persist to disk (under ./storage):
		index.storage_context.persist()

		# To reload from disk:
		from llama_index.core import StorageContext, load_index_from_storage

		# Rebuild storage context
		storage_context = StorageContext.from_defaults(persist_dir="./storage")
		# Load index
		index = load_index_from_storage(storage_context)

# REF [site] >> https://github.com/run-llama/llama-parse-py
def llama_parse_example():
	# Install:
	#	pip install llama_cloud

	if True:
		import os
		from llama_cloud import LlamaCloud

		client = LlamaCloud(
			api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),  # This is the default and can be omitted
		)

		parsing = client.parsing.create(
			tier="agentic",
			version="latest",
			file_id="abc1234",
		)
		print(parsing.id)

	if True:
		# Async usage
		#	Simply import AsyncLlamaCloud instead of LlamaCloud and use await with each API call.

		import os
		import asyncio
		from llama_cloud import AsyncLlamaCloud

		client = AsyncLlamaCloud(
			api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),  # This is the default and can be omitted
		)

		async def main() -> None:
			parsing = await client.parsing.create(
				tier="agentic",
				version="latest",
				file_id="abc1234",
			)
			print(parsing.id)

		asyncio.run(main())

# REF [site] >> https://developers.llamaindex.ai/liteparse/guides/library-usage/
def lite_parse_example():
	# Install:
	#	pip install liteparse

	from liteparse import LiteParse

	if True:
		parser = LiteParse()
	elif False:
		# Configuration
		parser = LiteParse(
			ocr_enabled=True,
			ocr_server_url="http://localhost:8828/ocr",
			ocr_language="fra",
			dpi=300,
			target_pages="1-5",
			password="secret",
		)
	elif False:
		# Markdown output
		parser = LiteParse(
			output_format="markdown",  # "json" | "text" | "markdown"
			image_mode="placeholder",  # "placeholder" | "off" | "embed"
			extract_links=True,  # render [text](url) link syntax (default: True)
		)

	if True:
		# Parse a document
		result = parser.parse("document.pdf")

		print(result.text)

		# Per-page data
		for page in result.pages:
			print(f"Page {page.page_num}: {len(page.text_items)} text items")
	elif False:
		# Parse from bytes
		with open("document.pdf", "rb") as f:
			pdf_bytes = f.read()

		result = parser.parse(pdf_bytes)
		print(result.text)
	elif False:
		# Screenshots
		screenshots = parser.screenshot("document.pdf", page_numbers=[1, 2, 3])
		for s in screenshots:
			print(f"Page {s.page_num}: {s.width}x{s.height}")
			# s.image_bytes contains PNG data

def main():
	#llama_index_example()  # LlamaIndex

	llama_parse_example()  # LlamaParse
	#lite_parse_example()  # LiteParse

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
