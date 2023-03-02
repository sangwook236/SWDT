#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://huggingface.co/docs/huggingface_hub/quick-start
def quickstart():
	import huggingface_hub

	# Download files.
	huggingface_hub.hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")

	huggingface_hub.hf_hub_download(
		repo_id="google/pegasus-xsum", 
		filename="config.json", 
		revision="4d33b01d79672f27f001f6abade33f22d993b151"
	)

	# Login.
	#	huggingface-cli login
	huggingface_hub.login()

	# Create a repository.
	api = huggingface_hub.HfApi()
	api.create_repo(repo_id="super-cool-model")
	api.create_repo(repo_id="super-cool-model", private=True)

	# Upload files.
	api = huggingface_hub.HfApi()
	api.upload_file(
		path_or_fileobj="/home/lysandre/dummy-test/README.md", 
		path_in_repo="README.md", 
		repo_id="lysandre/test-model",
	)

# REF [site] >> https://huggingface.co/docs/huggingface_hub/searching-the-hub
def searching_the_hub_guide():
	import huggingface_hub

	api = huggingface_hub.HfApi()

	models = api.list_models()
	datasets = api.list_datasets()
	spaces = api.list_spaces()

	print(f"#models = {len(list(iter(models)))}.")
	print(f"#datasets = {len(list(iter(datasets)))}.")
	print(f"#spaces = {len(list(iter(spaces)))}.")
	print(f"Model #0: {next(iter(models))}.")
	print(f"Dataset #0: {next(iter(datasets))}.")
	print(f"Space #0: {next(iter(spaces))}.")

	models = api.list_models(
		filter=huggingface_hub.ModelFilter(
			task="image-classification",
			library="pytorch",
			trained_dataset="imagenet"
		)
	)

	print(f"#models: {len(list(iter(models)))}.")
	print(f"Model #0: {next(iter(models))}.")

	datasets = api.list_datasets(sort="downloads", direction=-1, limit=5)

	print(f"#datasets = {len(list(iter(datasets)))}.")
	print(f"Dataset #0: {next(iter(datasets))}.")

	model_args = huggingface_hub.ModelSearchArguments()
	dataset_args = huggingface_hub.DatasetSearchArguments()

	print(f"model_args: {model_args}.")
	print(f"model_args.library: {model_args.library}.")
	print(f"model_args.library.PyTorch: {model_args.library.PyTorch}.")
	print(f"model_args.pipeline_tag.TextClassification: {model_args.pipeline_tag.TextClassification}.")
	print(f"model_args.dataset.glue: {model_args.dataset.glue}.")

	print(f"dataset_args: {dataset_args}.")

	#-----
	args = huggingface_hub.ModelSearchArguments()

	# List only the text classification models.
	api.list_models(filter="text-classification")
	# Using the 'ModelFilter'.
	filt = huggingface_hub.ModelFilter(task="text-classification")
	api.list_models(filter=filt)
	# With 'ModelSearchArguments'.
	filt = huggingface_hub.ModelFilter(task=args.pipeline_tags.TextClassification)
	api.list_models(filter=filt)

	# Using 'ModelFilter' and 'ModelSearchArguments' to find text classification in both PyTorch and TensorFlow.
	filt = huggingface_hub.ModelFilter(
		task=args.pipeline_tags.TextClassification,
		library=[args.library.PyTorch, args.library.TensorFlow],
	)
	api.list_models(filter=filt)

	# List only models from the AllenNLP library.
	api.list_models(filter="allennlp")
	# Using 'ModelFilter' and 'ModelSearchArguments'.
	filt = huggingface_hub.ModelFilter(library=args.library.allennlp)
	api.list_models(filter=filt)

	# Example usage with the 'search' argument:

	# List all models with "bert" in their name.
	api.list_models(search="bert")
	# List all models with "bert" in their name made by google.
	api.list_models(search="bert", author="google")

# REF [site] >> https://huggingface.co/docs/huggingface_hub/how-to-inference
def how_to_inference_guide():
	from huggingface_hub.inference_api import InferenceApi

	API_TOKEN = ...

	inference = InferenceApi(repo_id="bert-base-uncased", token=API_TOKEN)
	inference(inputs="The goal of life is [MASK].")

	inference = InferenceApi(repo_id="deepset/roberta-base-squad2", token=API_TOKEN)
	inputs = {"question": "Where is Hugging Face headquarters?", "context": "Hugging Face is based in Brooklyn, New York. There is also an office in Paris, France."}
	inference(inputs)

	inference = InferenceApi(repo_id="typeform/distilbert-base-uncased-mnli", token=API_TOKEN)
	inputs = "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
	params = {"candidate_labels":["refund", "legal", "faq"]}
	inference(inputs, params)

	inference = InferenceApi(
		repo_id="paraphrase-xlm-r-multilingual-v1", 
		task="feature-extraction", 
		token=API_TOKEN,
	)

# REF [site] >>
#	https://huggingface.co/blog/lora
#	https://huggingface.co/sayakpaul/sd-model-finetuned-lora-t4
def model_info_test():
	import huggingface_hub

	# LoRA weights ~3 MB.
	model_path = "sayakpaul/sd-model-finetuned-lora-t4"

	info = huggingface_hub.model_info(model_path)
	model_base = info.cardData["base_model"]
	print(model_base)  # Output: CompVis/stable-diffusion-v1-4.

def main():
	# Hugging Face Hub.

	#quickstart()
	searching_the_hub_guide()
	#how_to_inference_guide()

	#model_info_test()

	#--------------------
	# Hugging Face Transformers.
	#	Refer ./hugging_face_transformers_test.py.

	#--------------------
	# Hugging Face Diffusers.
	#	Refer ../machine_vision/diffusion_model_test.py.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
