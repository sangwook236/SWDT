#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://huggingface.co/docs/huggingface_hub/quick-start
def hub_quickstart():
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
def hub_searching_the_hub_guide():
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
def hub_how_to_inference_guide():
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
def hub_model_info_test():
	import huggingface_hub

	# LoRA weights ~3 MB.
	model_path = "sayakpaul/sd-model-finetuned-lora-t4"

	info = huggingface_hub.model_info(model_path)
	model_base = info.cardData["base_model"]
	print(model_base)  # Output: CompVis/stable-diffusion-v1-4.

# REF [site] >> https://github.com/huggingface/datasets
def datasets_quick_example():
	import transformers
	import datasets

	# Print all the available datasets.
	#print(datasets.list_datasets())
	print(f"#datasets = {len(datasets.list_datasets())}.")

	#-----
	# Load a dataset and print the first example in the training set.
	squad_dataset = datasets.load_dataset("squad")  # datasets.DatasetDict.
	assert isinstance(squad_dataset, datasets.DatasetDict)
	print("SQuAD:")
	print(f"Dataset keys: {squad_dataset.keys()}.")
	print(f"#train data = {len(squad_dataset['train'])}, #validation data = {len(squad_dataset['validation'])}.")
	print(f"Train example keys: {squad_dataset['train'][0].keys()}.")
	print(f"Validation example keys: {squad_dataset['validation'][0].keys()}.")
	print(f"Train example #0: {squad_dataset['train'][0]}.")

	squad_train_dataset = datasets.load_dataset("squad", split="train")  # datasets.Dataset.
	assert isinstance(squad_train_dataset, datasets.Dataset)
	print("SQuAD (train):")
	print(f"Train example keys: {squad_train_dataset[0].keys()}.")
	print(f"Train example #0: {squad_train_dataset[0]}.")

	# Process the dataset - add a column with the length of the context texts.
	dataset_with_length = squad_dataset.map(lambda x: {"length": len(x["context"])})
	print(f"Dataset keys: {dataset_with_length.keys()}.")
	print(f"Train example keys: {dataset_with_length['train'][0].keys()}.")
	print(f"Validation example keys: {squad_dataset['validation'][0].keys()}.")

	# Process the dataset - tokenize the context texts (using a tokenizer from the Hugging Face Transformers library).
	tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

	tokenized_dataset = squad_dataset.map(lambda x: tokenizer(x["context"]), batched=True)
	print(f"Dataset keys: {tokenized_dataset.keys()}.")
	print(f"Train example keys: {tokenized_dataset['train'][0].keys()}.")
	print(f"Validation example keys: {squad_dataset['validation'][0].keys()}.")

	#-----
	# If your dataset is bigger than your disk or if you don't want to wait to download the data, you can use streaming:

	# If you want to use the dataset immediately and efficiently stream the data as you iterate over the dataset.
	image_dataset = datasets.load_dataset("cifar100", streaming=True)
	print("CIFAR-100:")
	print(f"Keys: {image_dataset.keys()}.")
	#print(f"#train data = {len(image_dataset['train'])}, #test data = {len(image_dataset['test'])}.")  # TypeError: object of type 'IterableDataset' has no len().

	for example in image_dataset["train"]:
		break

# REF [site] >> https://huggingface.co/docs/datasets/quickstart
def datasets_quicktour():
	import torch, torchvision
	import transformers
	import datasets

	# Audio.
	if False:
		dataset = datasets.load_dataset("PolyAI/minds14", "en-US", split="train")
		print("Audio:")
		print(f"Train example keys: {dataset[0].keys()}.")
		print(f"Train example #0: {dataset[0]}.")

		#model = transformers.AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base")
		feature_extractor = transformers.AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

		dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
		print(dataset[0]["audio"])

		def preprocess_function(examples):
			audio_arrays = [x["array"] for x in examples["audio"]]
			inputs = feature_extractor(
				audio_arrays,
				sampling_rate=16000,
				padding=True,
				max_length=100000,
				truncation=True,
			)
			return inputs

		dataset = dataset.map(preprocess_function, batched=True)
		print(f"Train example keys: {dataset[0].keys()}.")
		dataset = dataset.rename_column("intent_class", "labels")
		print(f"Train example keys: {dataset[0].keys()}.")

		dataset.set_format(type="torch", columns=["input_values", "labels"])
		print(f"Train example keys: {dataset[0].keys()}.")
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

	#--------------------
	# Vision.
	if False:
		dataset = datasets.load_dataset("beans", split="train")
		print("Vision:")
		print(f"Train example keys: {dataset[0].keys()}.")
		print(f"Train example #0: {dataset[0]}.")

		jitter = torchvision.transforms.Compose([
			torchvision.transforms.ColorJitter(brightness=0.5, hue=0.5),
			torchvision.transforms.ToTensor()
		])

		def transforms(examples):
			examples["pixel_values"] = [jitter(image.convert("RGB")) for image in examples["image"]]
			return examples

		dataset = dataset.with_transform(transforms)
		print(f"Train example keys: {dataset[0].keys()}.")

		def collate_fn(examples):
			images = []
			labels = []
			for example in examples:
				images.append((example["pixel_values"]))
				labels.append(example["labels"])
				
			pixel_values = torch.stack(images)
			labels = torch.tensor(labels)
			return {"pixel_values": pixel_values, "labels": labels}
		dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=4)

	#--------------------
	# NLP.
	if True:
		dataset = datasets.load_dataset("glue", "mrpc", split="train")
		print("NLP:")
		print(f"Train example keys: {dataset[0].keys()}.")
		print(f"Train example #0: {dataset[0]}.")

		#model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
		tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

		def encode(examples):
			return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

		dataset = dataset.map(encode, batched=True)
		print(f"Train example keys: {dataset[0].keys()}.")
		print(f"Train example #0: {dataset[0]}.")

		dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)
		print(f"Train example keys: {dataset[0].keys()}.")

		dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
		print(f"Train example keys: {dataset[0].keys()}.")
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# REF [site] >> https://huggingface.co/docs/datasets/tutorial
def datasets_tutorials():
	import time
	import datasets

	#--------------------
	# Load a dataset from the Hub.
	#	https://huggingface.co/docs/datasets/load_hub
	if True:
		# Load a dataset.
		ds_builder = datasets.load_dataset_builder("rotten_tomatoes")

		# Inspect dataset description.
		print(f"Data description: {ds_builder.info.description}.")
		# Inspect dataset features.
		print(f"Dataset features: {ds_builder.info.features}.")

		dataset = datasets.load_dataset("rotten_tomatoes", split="train")

		# Splits.
		print(f"Dataset split names: {datasets.get_dataset_split_names('rotten_tomatoes')}.")

		dataset = datasets.load_dataset("rotten_tomatoes", split="train")
		print(dataset)

		dataset = datasets.load_dataset("rotten_tomatoes")
		print(dataset)

		# Configurations.
		configs = datasets.get_dataset_config_names("PolyAI/minds14")
		print(f"Configurations: {configs}.")

		mindsFR = datasets.load_dataset("PolyAI/minds14", "fr-FR", split="train")
		print(mindsFR)

	#--------------------
	# Know your dataset.
	#	https://huggingface.co/docs/datasets/access
	if False:
		# Dataset.

		dataset = datasets.load_dataset("rotten_tomatoes", split="train")

		# Indexing.
		print(dataset[0])
		print(dataset[-1])

		#print(dataset["text"])
		print(len(dataset["text"]))
		print(dataset[0]["text"])

		# It is important to remember that indexing order matters, especially when working with large audio and image datasets.
		# Indexing by the column name returns all the values in the column first, then loads the value at that position.
		# For large datasets, it may be slower to index by the column name first.
		start_time = time.time()
		dataset[0]['text']
		print(f"Elapsed time: {time.time() - start_time} secs.")

		start_time = time.time()
		dataset["text"][0]
		print(f"Elapsed time: {time.time() - start_time} secs.")

		# Slicing.
		print(dataset[:3])
		print(dataset[3:6])

		#-----
		# IterableDataset.

		iterable_dataset = datasets.load_dataset("food101", split="train", streaming=True)
		for example in iterable_dataset:
			print(example)
			break

		print(next(iter(iterable_dataset)))

		for example in iterable_dataset:
			print(example)
			break

		print(list(iterable_dataset.take(3)))

	#--------------------
	# Preprocess.
	#	https://huggingface.co/docs/datasets/use_dataset
	if False:
		# Tokenize text.
		import transformers

		tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
		dataset = datasets.load_dataset("rotten_tomatoes", split="train")

		print(f"tokenizer(dataset[0]['text']): {tokenizer(dataset[0]['text'])}.")

		def tokenization(example):
			return tokenizer(example["text"])

		dataset = dataset.map(tokenization, batched=True)

		dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
		print(f"dataset.format['type']: {dataset.format['type']}.")

		#-----
		# Resample audio signals.
		feature_extractor = transformers.AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
		dataset = datasets.load_dataset("PolyAI/minds14", "en-US", split="train")

		print(f"dataset[0]['audio']: {dataset[0]['audio']}.")

		dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
		print(f"dataset[0]['audio']: {dataset[0]['audio']}.")

		def preprocess_function(examples):
			audio_arrays = [x["array"] for x in examples["audio"]]
			inputs = feature_extractor(audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True)
			return inputs

		dataset = dataset.map(preprocess_function, batched=True)

		#-----
		# Apply data augmentations.
		import torchvision

		feature_extractor = transformers.AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
		dataset = datasets.load_dataset("beans", split="train")

		print(f"dataset[0]['image']: {dataset[0]['image']}.")

		rotate = torchvision.transforms.RandomRotation(degrees=(0, 90))
		def transforms(examples):
			examples["pixel_values"] = [rotate(image.convert("RGB")) for image in examples["image"]]
			return examples

		dataset.set_transform(transforms)
		print(f"dataset[0]['pixel_values']: {dataset[0]['pixel_values']}.")

	#--------------------
	# Evaluate predictions.
	#	https://huggingface.co/docs/datasets/metrics
	if False:
		metrics_list = datasets.list_metrics()
		print(f"Metrics: {metrics_list}.")
		print(f"#metircs = {len(metrics_list)}.")

		# Load metric.
		#	This will load the metric associated with the MRPC dataset from the GLUE benchmark.
		metric = datasets.load_metric("glue", "mrpc")

		# Select a configuration.
		metric = datasets.load_metric("glue", "mrpc")

		# Metrics object.
		print(f"Inputs description: {metric.inputs_description}.")

		# Compute metric.
		"""
		model_predictions = model(model_inputs)
		final_score = metric.compute(predictions=model_predictions, references=gold_references)
		print(f"Final score = {final_score}.")
		"""

	#--------------------
	# Create a dataset.
	#	https://huggingface.co/docs/datasets/create_dataset
	if False:
		# Folder-based builders.
		dataset = datasets.load_dataset("imagefolder", data_dir="/path/to/pokemon")
		dataset = datasets.load_dataset("audiofolder", data_dir="/path/to/folder")

		# From local files.
		def gen():
			yield {"pokemon": "bulbasaur", "type": "grass"}
			yield {"pokemon": "squirtle", "type": "water"}

		ds = datasets.Dataset.from_generator(gen)
		print(ds[0])

		ds = datasets.IterableDataset.from_generator(gen)
		for example in ds:
			print(example)

		ds = datasets.Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"], "type": ["grass", "water"]})
		print(ds[0])

		#audio_dataset = datasets.Dataset.from_dict({"audio": ["path/to/audio_1", ..., "path/to/audio_n"]}).cast_column("audio", datasets.Audio())

	#--------------------
	# Share a dataset to the Hub.
	#	https://huggingface.co/docs/datasets/upload_dataset

	# Upload with the Hub UI.
	#	Create a repository.
	#	Upload dataset.
	#	Create a Dataset card.
	#	Load dataset.

	# Upload with Python.

# REF [site] >> https://huggingface.co/docs/datasets/how_to
def datasets_how_to_guides__general_use():
	import torch
	import datasets

	# General usage:
	#	Functions for general dataset loading and processing.
	#	The functions shown in this section are applicable across all dataset modalities.

	#--------------------
	# Load.

	# Hugging Face Hub.

	dataset = datasets.load_dataset("lhoestq/demo1")

	dataset = datasets.load_dataset(
		"lhoestq/custom_squad",
		revision="main"  # Tag name, or branch name, or commit hash.
	)

	data_files = {"train": "train.csv", "test": "test.csv"}
	dataset = datasets.load_dataset("namespace/your_dataset_name", data_files=data_files)

	c4_subset = datasets.load_dataset("allenai/c4", data_files="en/c4-train.0000*-of-01024.json.gz")
	c4_subset = datasets.load_dataset("allenai/c4", data_dir="en")

	data_files = {"validation": "en/c4-validation.*.json.gz"}
	c4_validation = datasets.load_dataset("allenai/c4", data_files=data_files, split="validation")

	#-----
	# Local loading script.

	dataset = datasets.load_dataset("path/to/local/loading_script/loading_script.py", split="train")
	dataset = datasets.load_dataset("path/to/local/loading_script", split="train")  # Equivalent because the file has the same name as the directory.

	# Edit loading script.
	#	git clone https://huggingface.co/datasets/eli5
	eli5 = datasets.load_dataset("path/to/local/eli5")

	#-----
	# Local and remote files.

	# CSV.
	dataset = datasets.load_dataset("csv", data_files="my_file.csv")

	# JSON.
	dataset = datasets.load_dataset("json", data_files="my_file.json")
	dataset = datasets.load_dataset("json", data_files="my_file.json", field="data")

	# Load remote JSON files via HTTP.
	base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
	dataset = datasets.load_dataset("json", data_files={"train": base_url + "train-v1.1.json", "validation": base_url + "dev-v1.1.json"}, field="data")

	# Parquet.
	dataset = datasets.load_dataset("parquet", data_files={"train": "train.parquet", "test": "test.parquet"})

	# Load remote Parquet files via HTTP.
	base_url = "https://storage.googleapis.com/huggingface-nlp/cache/datasets/wikipedia/20200501.en/1.0.0/"
	data_files = {"train": base_url + "wikipedia-train.parquet"}
	wiki = datasets.load_dataset("parquet", data_files=data_files, split="train")

	# SQL.
	dataset = datasets.Dataset.from_sql("data_table_name", con="sqlite:///sqlite_file.db")
	dataset = datasets.Dataset.from_sql("SELECT text FROM table WHERE length(text) > 100 LIMIT 10", con="sqlite:///sqlite_file.db")

	# Multiprocessing.
	oscar_afrikaans = datasets.load_dataset("oscar-corpus/OSCAR-2201", "af", num_proc=8)
	imagenet = datasets.load_dataset("imagenet-1k", num_proc=8)
	ml_librispeech_spanish = datasets.load_dataset("facebook/multilingual_librispeech", "spanish", num_proc=8)

	# In-memory data.

	# Python dictionary.
	my_dict = {"a": [1, 2, 3]}
	dataset = datasets.Dataset.from_dict(my_dict)

	# Python list of dictionaries.
	my_list = [{"a": 1}, {"a": 2}, {"a": 3}]
	dataset = datasets.Dataset.from_list(my_list)

	# Python generator.
	def my_gen():
		for i in range(1, 4):
			yield {"a": i}
	dataset = datasets.Dataset.from_generator(my_gen)

	def gen(shards):
		for shard in shards:
			with open(shard) as f:
				for line in f:
					yield {"line": line}
	shards = [f"data{i}.txt" for i in range(32)]
	ds = datasets.IterableDataset.from_generator(gen, gen_kwargs={"shards": shards})
	ds = ds.shuffle(seed=42, buffer_size=10_000)  # Shuffles the shards order + uses a shuffle buffer.
	dataloader = torch.utils.data.DataLoader(ds.with_format("torch"), num_workers=4)  # Give each worker a subset of 32/4=8 shards.

	# Pandas DataFrame.
	import pandas as pd
	df = pd.DataFrame({"a": [1, 2, 3]})
	dataset = datasets.Dataset.from_pandas(df)

	#-----
	# Slice splits.

	# Concatenate a train and test split by:
	train_test_ds = datasets.load_dataset("bookcorpus", split="train+test")
	# Select specific rows of the train split:
	train_10_20_ds = datasets.load_dataset("bookcorpus", split="train[10:20]")
	# Select a percentage of a split with:
	train_10pct_ds = datasets.load_dataset("bookcorpus", split="train[:10%]")
	# Select a combination of percentages from each split:
	train_10_80pct_ds = datasets.load_dataset("bookcorpus", split="train[:10%]+train[-80%:]")
	# Create cross-validated splits.
	val_ds = datasets.load_dataset("bookcorpus", split=[f"train[{k}%:{k+10}%]" for k in range(0, 100, 10)])
	train_ds = datasets.load_dataset("bookcorpus", split=[f"train[:{k}%]+train[{k+10}%:]" for k in range(0, 100, 10)])

	# Percent slicing and rounding.
	train_50_52_ds = datasets.load_dataset("bookcorpus", split="train[50%:52%]")
	train_52_54_ds = datasets.load_dataset("bookcorpus", split="train[52%:54%]")

	train_50_52pct1_ds = datasets.load_dataset("bookcorpus", split=datasets.ReadInstruction("train", from_=50, to=52, unit="%", rounding="pct1_dropremainder"))
	train_52_54pct1_ds = datasets.load_dataset("bookcorpus", split=datasets.ReadInstruction("train", from_=52, to=54, unit="%", rounding="pct1_dropremainder"))
	train_50_52pct1_ds = datasets.load_dataset("bookcorpus", split="train[50%:52%](pct1_dropremainder)")
	train_52_54pct1_ds = datasets.load_dataset("bookcorpus", split="train[52%:54%](pct1_dropremainder)")

	#--------------------
	# Process.

	dataset = datasets.load_dataset("glue", "mrpc", split="train")

	#-----
	# Sort, shuffle, select, split, and shard.

	# Sort.
	print(f"dataset['label'][:10]: {dataset['label'][:10]}.")
	sorted_dataset = dataset.sort("label")
	print(f"sorted_dataset['label'][:10]: {sorted_dataset['label'][:10]}.")
	print(f"sorted_dataset['label'][-10:]: {sorted_dataset['label'][-10:]}.")

	# Shuffle.
	shuffled_dataset = sorted_dataset.shuffle(seed=42)
	print(f"shuffled_dataset['label'][:10]: {shuffled_dataset['label'][:10]}.")

	iterable_dataset = dataset.to_iterable_dataset(num_shards=128)
	shuffled_iterable_dataset = iterable_dataset.shuffle(seed=42, buffer_size=1000)

	# Select and filter.
	small_dataset = dataset.select([0, 10, 20, 30, 40, 50])
	len(small_dataset)

	start_with_ar = dataset.filter(lambda example: example["sentence1"].startswith("Ar"))
	len(start_with_ar)
	print(f"start_with_ar['sentence1']: {start_with_ar['sentence1']}.")

	even_dataset = dataset.filter(lambda example, idx: idx % 2 == 0, with_indices=True)
	print(f"#even dataset = {len(even_dataset)}.")
	print(f"#dataset / 2 = {len(dataset) / 2}.")

	# Split.
	dataset.train_test_split(test_size=0.1)
	print(f"0.1 * len(dataset) = {0.1 * len(dataset)}.")

	# Shard.
	datasets = dataset.load_dataset("imdb", split="train")
	print(f"Dataset: {dataset}.")

	print(f"dataset.shard(num_shards=4, index=0): {dataset.shard(num_shards=4, index=0)}.")

	#-----
	# Rename, remove, cast, and flatten.

	# Rename.
	print(f"Dataset: {dataset}.")
	dataset = dataset.rename_column("sentence1", "sentenceA")
	dataset = dataset.rename_column("sentence2", "sentenceB")
	print(f"Dataset: {dataset}.")

	# Remove.
	dataset = dataset.remove_columns("label")
	print(f"dataset: {dataset}.")
	dataset = dataset.remove_columns(["sentence1", "sentence2"])
	print(f"dataset: {dataset}.")

	# Cast.
	print(f"dataset.features: {dataset.features}.")

	new_features = dataset.features.copy()
	new_features["label"] = datasets.ClassLabel(names=["negative", "positive"])
	new_features["idx"] = datasets.Value("int64")
	dataset = dataset.cast(new_features)
	print(f"dataset.features: {dataset.features}.")

	dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
	print(f"dataset.features: {dataset.features}.")

	# Flatten.
	dataset = datasets.load_dataset("squad", split="train")
	print(f"dataset.features: {dataset.features}.")

	flat_dataset = dataset.flatten()
	print(f"flat_dataset: {flat_dataset}.")

	#-----
	# Map.

	def add_prefix(example):
		example["sentence1"] = "My sentence: " + example["sentence1"]
		return example

	updated_dataset = small_dataset.map(add_prefix)
	print(f"updated_dataset['sentence1'][:5]: {updated_dataset['sentence1'][:5]}.")

	updated_dataset = dataset.map(lambda example: {"new_sentence": example["sentence1"]}, remove_columns=["sentence1"])
	print(f"updated_dataset.column_names: {updated_dataset.column_names}.")

	updated_dataset = dataset.map(lambda example, idx: {"sentence2": f"{idx}: " + example["sentence2"]}, with_indices=True)
	print(f"updated_dataset['sentence2'][:5]: {updated_dataset['sentence2'][:5]}.")

	from multiprocess import set_start_method
	import torch
	import os

	set_start_method("spawn")

	def gpu_computation(example, rank):
		os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())
		# Your big GPU call goes here.
		return example
	updated_dataset = dataset.map(gpu_computation, with_rank=True)

	# Multiprocessing.
	updated_dataset = dataset.map(lambda example, idx: {"sentence2": f"{idx}: " + example["sentence2"]}, num_proc=4)

	# Batch processing.

	# Split long examples.
	def chunk_examples(examples):
		chunks = []
		for sentence in examples["sentence1"]:
			chunks += [sentence[i:i + 50] for i in range(0, len(sentence), 50)]
		return {"chunks": chunks}
	chunked_dataset = dataset.map(chunk_examples, batched=True, remove_columns=dataset.column_names)
	print(f"chunked_dataset[:10]: {chunked_dataset[:10]}.")
	print(f"dataset: {dataset}.")
	print(f"chunked_dataset: {chunked_dataset}.")

	# Data augmentation.
	import random
	import transformers

	fillmask = transformers.pipeline("fill-mask", model="roberta-base")
	mask_token = fillmask.tokenizer.mask_token
	smaller_dataset = dataset.filter(lambda e, i: i < 100, with_indices=True)

	def augment_data(examples):
		outputs = []
		for sentence in examples["sentence1"]:
			words = sentence.split(' ')
			K = random.randint(1, len(words)-1)
			masked_sentence = " ".join(words[:K]  + [mask_token] + words[K+1:])
			predictions = fillmask(masked_sentence)
			augmented_sequences = [predictions[i]["sequence"] for i in range(3)]
			outputs += [sentence] + augmented_sequences
		return {"data": outputs}

	augmented_dataset = smaller_dataset.map(augment_data, batched=True, remove_columns=dataset.column_names, batch_size=8)
	print(f"augmented_dataset[:9]['data']: {augmented_dataset[:9]['data']}.")

	# Process multiple splits.
	dataset = datasets.load_dataset("glue", "mrpc")
	encoded_dataset = dataset.map(lambda examples: tokenizer(examples["sentence1"]), batched=True)
	print(f"encoded_dataset['train'][0]: {encoded_dataset['train'][0]}.")

	# Distributed usage.
	dataset1 = datasets.Dataset.from_dict({"a": [0, 1, 2]})

	if training_args.local_rank > 0:
		print("Waiting for main process to perform the mapping")
		torch.distributed.barrier()

	dataset2 = dataset1.map(lambda x: {"a": x["a"] + 1})

	if training_args.local_rank == 0:
		print("Loading results from main process")
		torch.distributed.barrier()

	#-----
	# Concatenate.
	bookcorpus = datasets.load_dataset("bookcorpus", split="train")
	wiki = datasets.load_dataset("wikipedia", "20220301.en", split="train")
	wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # Only keep the 'text' column.

	assert bookcorpus.features.type == wiki.features.type
	bert_dataset = datasets.concatenate_datasets([bookcorpus, wiki])

	bookcorpus_ids = datasets.Dataset.from_dict({"ids": list(range(len(bookcorpus)))})
	bookcorpus_with_ids = datasets.concatenate_datasets([bookcorpus, bookcorpus_ids], axis=1)

	# Interleave.
	seed = 42
	probabilities = [0.3, 0.5, 0.2]
	d1 = datasets.Dataset.from_dict({"a": [0, 1, 2]})
	d2 = datasets.Dataset.from_dict({"a": [10, 11, 12, 13]})
	d3 = datasets.Dataset.from_dict({"a": [20, 21, 22]})
	dataset = datasets.interleave_datasets([d1, d2, d3], probabilities=probabilities, seed=seed)
	print(f"dataset['a']: {dataset['a']}.")

	d1 = Dataset.from_dict({"a": [0, 1, 2]})
	d2 = Dataset.from_dict({"a": [10, 11, 12, 13]})
	d3 = Dataset.from_dict({"a": [20, 21, 22]})
	dataset = datasets.interleave_datasets([d1, d2, d3], stopping_strategy="all_exhausted")
	print(f"dataset['a']: {dataset['a']}.")

	#-----
	# Format.

	dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
	dataset = dataset.with_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
	print(f"dataset.format: {dataset.format}.")
	dataset.reset_format()
	print(f"dataset.format: {dataset.format}.")

	# Format transform.
	from transformers import AutoTokenizer
	tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
	def encode(batch):
		return tokenizer(batch["sentence1"], padding="longest", truncation=True, max_length=512, return_tensors="pt")
	dataset.set_transform(encode)
	print(f"dataset.format: {dataset.format}.")

	#-----
	# Save.
	encoded_dataset.save_to_disk("/path/of/my/dataset/directory")
	reloaded_dataset = datasets.load_from_disk("/path/of/my/dataset/directory")

	# Export.
	encoded_dataset.to_csv("/path/ot/my/dataset.csv")
	#encoded_dataset.to_json("/path/ot/my/dataset.json")
	#encoded_dataset.to_parquet("/path/ot/my/dataset.parquet")
	#encoded_dataset.to_sql("/path/ot/my/dataset.sql")

def datasets_how_to_guides__audio():
	# Audio:
	#	How to load, process, and share audio datasets.

	raise NotImplementedError

def datasets_how_to_guides__vision():
	# Vision:
	#	How to load, process, and share image datasets.

	raise NotImplementedError

def datasets_how_to_guides__text():
	# Text:
	#	How to load, process, and share text datasets.

	raise NotImplementedError

def datasets_how_to_guides__tabular():
	# Tabular:
	#	How to load, process, and share tabular datasets.

	raise NotImplementedError

def datasets_how_to_guides__dataset_repository():
	# Dataset repository:
	#	How to share and upload a dataset to the Hub.

	raise NotImplementedError

def datasets_hugging_face_test():
	import datasets

	if False:
		# Wikipedia.
		#	REF [site] >> https://huggingface.co/datasets/wikipedia
		
		#ds = datasets.load_dataset("wikipedia", language="sw", date="20220120", beam_runner=...)
		ds = datasets.load_dataset("wikipedia", "20220301.en")  # ~20.3GB.

	if True:
		# TextVQA.
		#	REF [site] >> https://huggingface.co/datasets/textvqa

		dataset_name = "textvqa"  # ~8.04GB.
		split = "test"  # {None, 'train', 'validation', 'test'}.
		example_idx = 0

		#ds_dict = datasets.load_dataset(dataset_name)  # datasets.DatasetDict.
		#ds = ds_dict[split]
		ds = datasets.load_dataset(dataset_name, split=split)  # datasets.Dataset.

		print(f"{ds=}.")
		print(f'{ds.features["image_id"]=}.')  # datasets.Value.
		print(f'{ds.features["question_id"]=}.')  # datasets.Value.
		print(f'{ds.features["question"]=}.')  # datasets.Value.
		print(f'{ds.features["question_tokens"]=}.')  # datasets.Sequence.
		print(f'{ds.features["image"]=}.')  # datasets.Image.
		print(f'{ds.features["image_width"]=}.')  # datasets.Value.
		print(f'{ds.features["image_height"]=}.')  # datasets.Value.
		print(f'{ds.features["flickr_original_url"]=}.')  # datasets.Value.
		print(f'{ds.features["flickr_300k_url"]=}.')  # datasets.Value.
		print(f'{ds.features["answers"]=}.')  # datasets.Sequence.
		print(f'{ds.features["image_classes"]=}.')  # datasets.Sequence.
		print(f'{ds.features["set_name"]=}.')  # datasets.Value.
		print(f"#examples = {ds.num_rows}.")
		assert example_idx < ds.num_rows

		example = ds[example_idx]
		print(f'{example["image_id"]=}.')  # str.
		print(f'{example["question_id"]=}.')  # int32.
		print(f'{example["question"]=}.')  # str.
		print(f'{example["question_tokens"]=}.')  # list of strs.
		print(f'{example["image"]=}.')  # PIL.Image.
		print(f'{example["image_width"]=}.')  # int32.
		print(f'{example["image_height"]=}.')  # int32.
		print(f'{example["flickr_original_url"]=}.')  # str.
		print(f'{example["flickr_300k_url"]=}.')  # str.
		print(f'{example["answers"]=}.')  # list of strs.
		print(f'{example["image_classes"]=}.')  # list of strs.
		print(f'{example["set_name"]=}.')  # str.

	if True:
		# WikiSQL.
		#	REF [site] >> https://huggingface.co/datasets/wikisql

		dataset_name = "wikisql"  # ~26.2MB.
		split = "test"  # {None, 'train', 'validation', 'test'}.
		example_idx = 0

		#ds_dict = datasets.load_dataset(dataset_name)  # datasets.DatasetDict.
		#ds = ds_dict[split]
		ds = datasets.load_dataset(dataset_name, split=split)  # datasets.Dataset.

		print(f"{ds=}.")
		print(f'{ds.features["phase"]=}.')  # datasets.Value.
		print(f'{ds.features["question"]=}.')  # datasets.Value.
		print(f'{ds.features["table"]=}.')  # dict. {'header', 'page_title', 'page_id', 'types', 'id', 'section_title', 'caption', 'rows', 'name'}.
		print(f'{ds.features["sql"]=}.')  # dict. {'human_readable', 'sel', 'agg', 'conds'}.
		print(f"#examples = {ds.num_rows}.")
		assert example_idx < ds.num_rows

		example = ds[example_idx]
		print(f'{example["phase"]=}.')  # int32.
		print(f'{example["question"]=}.')  # str.
		print(f'{example["table"]["header"]=}.')  # list of strs.
		print(f'{example["table"]["page_title"]=}.')  # str.
		print(f'{example["table"]["page_id"]=}.')  # str.
		print(f'{example["table"]["types"]=}.')  # list of strs.
		print(f'{example["table"]["id"]=}.')  # str.
		print(f'{example["table"]["section_title"]=}.')  # str.
		print(f'{example["table"]["caption"]=}.')  # str.
		print(f'{example["table"]["rows"]=}.')  # list of lists of strs.
		print(f'{example["table"]["name"]=}.')  # str.
		print(f'{example["sql"]["human_readable"]=}.')  # str.
		print(f'{example["sql"]["sel"]=}.')  # int.
		print(f'{example["sql"]["agg"]=}.')  # int.
		print(f'{example["sql"]["conds"]=}.')  # dict. {'column_index', 'operator_index', 'condition'}.

	if True:
		# WikiTQ.
		#	REF [site] >> https://huggingface.co/datasets/wikitablequestions

		dataset_name = "wikitablequestions"  # ~29.3MB.
		split = "test"  # {None, 'train', 'validation', 'test'}.
		example_idx = 0

		#ds_dict = datasets.load_dataset(dataset_name)  # datasets.DatasetDict.
		#ds = ds_dict[split]
		ds = datasets.load_dataset(dataset_name, split=split)  # datasets.Dataset.

		print(f"{ds=}.")
		print(f'{ds.features["id"]=}.')  # datasets.Value.
		print(f'{ds.features["question"]=}.')  # datasets.Value.
		print(f'{ds.features["answers"]=}.')  # datasets.Sequence.
		print(f'{ds.features["table"]=}.')  # dict. {'header', 'rows', 'name'}.
		print(f"#examples = {ds.num_rows}.")
		assert example_idx < ds.num_rows

		example = ds[example_idx]
		print(f'{example["id"]=}.')  # str.
		print(f'{example["question"]=}.')  # str.
		print(f'{example["answers"]=}.')  # list of strs.
		print(f'{example["table"]["header"]=}.')  # list of strs.
		print(f'{example["table"]["rows"]=}.')  # list of lists of strs.
		print(f'{example["table"]["name"]=}.')  # str.

# REF [site] >> https://huggingface.co/HuggingFaceM4
def datasets_hugging_face_m4_test():
	import datasets

	if True:
		# VQAv2.

		dataset_name = "HuggingFaceM4/VQAv2"  # ~20.2GB.
		split = "test"  # {None, 'train', 'validation', 'testdev', 'test'}.
		example_idx = 0

		#ds_dict = datasets.load_dataset(dataset_name)  # datasets.DatasetDict.
		#ds = ds_dict[split]
		ds = datasets.load_dataset(dataset_name, split=split)  # datasets.Dataset.

		print(f"{ds=}.")
		print(f'{ds.features["question_type"]=}.')  # datasets.Value.
		print(f'{ds.features["multiple_choice_answer"]=}.')  # datasets.Value.
		print(f'{ds.features["answers"]=}.')  # list of dicts.
		print(f'{ds.features["image_id"]=}.')  # datasets.Value.
		print(f'{ds.features["answer_type"]=}.')  # datasets.Value.
		print(f'{ds.features["question_id"]=}.')  # datasets.Value.
		print(f'{ds.features["question"]=}.')  # datasets.Value.
		print(f'{ds.features["image"]=}.')  # datasets.Image.
		print(f"#examples = {ds.num_rows}.")
		assert example_idx < ds.num_rows

		example = ds[example_idx]
		print(f'{example["question_type"]=}.')  # str.
		print(f'{example["multiple_choice_answer"]=}.')  # str.
		print(f'{example["answers"]=}.')  # list of dicts. {'answer', 'answer_confidence', 'answer_id'}.
		#print(f'{example["answers"][0]["answer"]=}.')
		#print(f'{example["answers"][0]["answer_confidence"]=}.')
		#print(f'{example["answers"][0]["answer_id"]=}.')
		print(f'{example["image_id"]=}.')  # int64.
		print(f'{example["answer_type"]=}.')  # str.
		print(f'{example["question_id"]=}.')  # int64.
		print(f'{example["question"]=}.')  # str.
		print(f'{example["image"]=}.')  # PIL.Image.

	if True:
		# VaTeX.

		dataset_name = "HuggingFaceM4/vatex"
		if False:
			subset = "v1.0"
			split = "public_test"  # {None, 'train', 'validation', 'public_test'}.
		else:
			subset = "v1.1"
			split = "public_test"  # {None, 'train', 'validation', 'public_test', 'private_test'}.
		example_idx = 0

		#ds_dict = datasets.load_dataset(dataset_name, subset)  # datasets.DatasetDict.
		#ds = ds_dict[split]
		ds = datasets.load_dataset(dataset_name, subset, split=split)  # datasets.Dataset.

		print(f"{ds=}.")
		print(f'{ds.features["videoID"]=}.')  # datasets.Value.
		print(f'{ds.features["path"]=}.')  # datasets.Value.
		print(f'{ds.features["start"]=}.')  # datasets.Value.
		print(f'{ds.features["end"]=}.')  # datasets.Value.
		print(f'{ds.features["enCap"]=}.')  # datasets.Sequence.
		print(f'{ds.features["chCap"]=}.')  # datasets.Sequence.
		print(f"#examples = {ds.num_rows}.")
		assert example_idx < ds.num_rows

		example = ds[example_idx]
		print(f'{example["videoID"]=}.')  # str.
		print(f'{example["path"]=}.')  # str.
		print(f'{example["start"]=}.')  # int32.
		print(f'{example["end"]=}.')  # int32.
		print(f'{example["enCap"]=}.')  # list of strs.
		print(f'{example["chCap"]=}.')  # list of strs.

	if True:
		# TextCaps.

		dataset_name = "HuggingFaceM4/TextCaps"  # ~181MB.
		split = "test"  # {None, 'train', 'validation', 'test'}.
		example_idx = 0

		#ds_dict = datasets.load_dataset(dataset_name)  # datasets.DatasetDict.
		#ds = ds_dict[split]
		ds = datasets.load_dataset(dataset_name, split=split)  # datasets.Dataset.

		print(f"{ds=}.")
		print(f'{ds.features["ocr_tokens"]=}.')  # list of datasets.Value's.
		print(f'{ds.features["ocr_info"]=}.')  # list of dicts.
		print(f'{ds.features["image"]=}.')  # datasets.Image.
		print(f'{ds.features["image_id"]=}.')  # datasets.Value.
		print(f'{ds.features["image_classes"]=}.')  # datasets.Value.
		print(f'{ds.features["flickr_original_url"]=}.')  # datasets.Value.
		print(f'{ds.features["flickr_300k_url"]=}.')  # datasets.Value.
		print(f'{ds.features["image_width"]=}.')  # datasets.Value.
		print(f'{ds.features["image_height"]=}.')  # datasets.Value.
		print(f'{ds.features["set_name"]=}.')  # datasets.Value.
		print(f'{ds.features["image_name"]=}.')  # datasets.Value.
		print(f'{ds.features["image_path"]=}.')  # datasets.Value.
		print(f'{ds.features["reference_strs"]=}.')  # datasets.Value.
		print(f'{ds.features["reference_tokens"]=}.')  # datasets.Value.
		print(f"#examples = {ds.num_rows}.")
		assert example_idx < ds.num_rows

		example = ds[example_idx]
		print(f'{example["ocr_tokens"]=}.')  # list of strs.
		print(f'{example["ocr_info"]=}.')  # list of dicts. {'word', 'bounding_box'}.
		print(f'{example["ocr_info"][0]["word"]=}.')  # str.
		print(f'{example["ocr_info"][0]["bounding_box"]=}.')  # dict. {'width', 'height', 'rotation', 'roll', 'pitch', 'yaw', 'top_left_x', 'top_left_y'}.
		print(f'{example["image"]=}.')  # PIL.Image.
		print(f'{example["image_id"]=}.')  # str.
		print(f'{example["image_classes"]=}.')  # list of strs.
		print(f'{example["flickr_original_url"]=}.')  # str.
		print(f'{example["flickr_300k_url"]=}.')  # str.
		print(f'{example["image_width"]=}.')  # int32.
		print(f'{example["image_height"]=}.')  # int32.
		print(f'{example["set_name"]=}.')  # str.
		print(f'{example["image_name"]=}.')  # str.
		print(f'{example["image_path"]=}.')  # str.
		print(f'{example["reference_strs"]=}.')  # str.
		print(f'{example["reference_tokens"]=}.')  # str.

	if True:
		# NoCaps.

		dataset_name = "HuggingFaceM4/NoCaps"
		split = "test"  # {None, 'validation', 'test'}.
		example_idx = 0

		#ds_dict = datasets.load_dataset(dataset_name)  # datasets.DatasetDict.
		#ds = ds_dict[split]
		ds = datasets.load_dataset(dataset_name, split=split)  # datasets.Dataset.

		print(f"{ds=}.")
		print(f'{ds.features["image"]=}.')  # datasets.Image.
		print(f'{ds.features["image_coco_url"]=}.')  # datasets.Value.
		print(f'{ds.features["image_date_captured"]=}.')  # datasets.Value.
		print(f'{ds.features["image_file_name"]=}.')  # datasets.Value.
		print(f'{ds.features["image_height"]=}.')  # datasets.Value.
		print(f'{ds.features["image_width"]=}.')  # datasets.Value.
		print(f'{ds.features["image_id"]=}.')  # datasets.Value.
		print(f'{ds.features["image_license"]=}.')  # datasets.Value.
		print(f'{ds.features["image_open_images_id"]=}.')  # datasets.Value.
		print(f'{ds.features["annotations_ids"]=}.')  # datasets.=Sequence.
		print(f'{ds.features["annotations_captions"]=}.')  # datasets.=Sequence.

		example = ds[example_idx]
		print(f'{example["image"]=}.')  # PIL.Image.
		print(f'{example["image_coco_url"]=}.')  # str.
		print(f'{example["image_date_captured"]=}.')  # str.
		print(f'{example["image_file_name"]=}.')  # str.
		print(f'{example["image_height"]=}.')  # int32.
		print(f'{example["image_width"]=}.')  # int32.
		print(f'{example["image_id"]=}.')  # int32.
		print(f'{example["image_license"]=}.')  # int8.
		print(f'{example["image_open_images_id"]=}.')  # str.
		print(f'{example["annotations_ids"]=}.')  # list of int32's.
		print(f'{example["annotations_captions"]=}.')  # list of strs.

# REF [site] >> https://huggingface.co/nielsr
def datasets_nielsr_test():
	from PIL import Image
	import matplotlib.pyplot as plt
	import datasets

	# FUNSD.

	dataset_name = "nielsr/funsd"  # ~16.8MB.
	split = "test"  # {None, 'train', 'test'}.
	example_idx = 0

	#ds_dict = datasets.load_dataset(dataset_name)  # datasets.DatasetDict.
	#ds = ds_dict[split]
	ds = datasets.load_dataset(dataset_name, split=split)  # datasets.Dataset.

	print(f"{ds=}.")
	"""
	DatasetDict({
		train: Dataset({
			features: ['id', 'words', 'bboxes', 'ner_tags', 'image_path'],
			num_rows: 149
		})
		test: Dataset({
			features: ['id', 'words', 'bboxes', 'ner_tags', 'image_path'],
			num_rows: 50
		})
	})
	"""

	print(f'{ds.features["id"]=}.')  # datasets.Value.
	print(f'{ds.features["words"]=}.')  # datasets.Sequence.
	print(f'{ds.features["bboxes"]=}.')  # datasets.Sequence.
	print(f'{ds.features["ner_tags"]=}.')  # datasets.Sequence(datasets.ClassLabel).
	print(f'{ds.features["image_path"]=}.')  # datasets.Value.
	print(f"#examples = {ds.num_rows}.")
	assert example_idx < ds.num_rows

	example = ds[example_idx]
	print(f'{example["id"]=}.')  # str.
	print(f'{example["words"]=}.')  # list of strs.
	print(f'{example["bboxes"]=}.')  # list of [x1, y1, x2, y2].
	# NER tags: {'O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER'}.
	print(f'{example["ner_tags"]=}.')  # list of ints.
	print(f'{example["image_path"]=}.')  # str.
	assert len(example["words"]) == len(example["bboxes"]) == len(example["ner_tags"])

	if False:
		# Visualize.
		image_path = example["image_path"]
		try:
			img = Image.open(image_path)

			#img.show()
			#plt.show()
		except IOError as ex:
			print(f"Failed to load an image, {image_path}: {ex}.")

# REF [site] >> https://huggingface.co/naver-clova-ix
def datasets_naver_clova_test():
	import json
	import matplotlib.pyplot as plt
	import datasets

	if True:
		# CORD.

		#dataset_name = "naver-clova-ix/cord-v1"
		dataset_name = "naver-clova-ix/cord-v2"  # ~2.3GB.
		split = "test"  # {None, 'train', 'validation', 'test'}.
		example_idx = 0

		#ds_dict = datasets.load_dataset(dataset_name)  # datasets.DatasetDict.
		#ds = ds_dict[split]
		ds = datasets.load_dataset(dataset_name, split=split)  # datasets.Dataset.

		print(f"{ds=}.")
		"""
		DatasetDict({
			train: Dataset({
				features: ['image', 'ground_truth'],
				num_rows: 800
			})
			validation: Dataset({
				features: ['image', 'ground_truth'],
				num_rows: 100
			})
			test: Dataset({
				features: ['image', 'ground_truth'],
				num_rows: 100
			})
		})
		"""

		print(f'{ds.features["image"]=}.')  # datasets.Image.
		print(f'{ds.features["ground_truth"]=}.')  # datasets.Value.
		print(f"#examples = {ds.num_rows}.")
		assert example_idx < ds.num_rows

		example = ds[example_idx]
		print(f'{example["image"]=}.')  # PIL.Image.
		print(f'{example["ground_truth"]=}.')  # str. JSON format.

		if False:
			# Visualize.
			img = example["image"]
			gt = json.loads(example["ground_truth"])

			print("G/T:")
			print(gt)
			img.show()
			plt.show()

		ds_cast = ds.cast_column("image", datasets.Image(decode=False))
		print(f'{ds_cast.features["image"]=}.')  # datasets.Image.
		print(f'{ds.features["image"]=}.')  # datasets.Image.

	if True:
		# SynthDoG.

		dataset_name = "naver-clova-ix/synthdog-en"  # ~42GB.
		#dataset_name = "naver-clova-ix/synthdog-ko"
		#dataset_name = "naver-clova-ix/synthdog-zh"
		#dataset_name = "naver-clova-ix/synthdog-ja"
		split = "validation"  # {None, 'train', 'validation'}.
		example_idx = 0

		#ds_dict = datasets.load_dataset(dataset_name)  # datasets.DatasetDict.
		#ds = ds_dict[split]
		ds = datasets.load_dataset(dataset_name, split=split)  # datasets.Dataset.

		print(f"{ds=}.")
		print(f'{ds.features["image"]=}.')  # datasets.Image.
		print(f'{ds.features["ground_truth"]=}.')  # datasets.Value.
		print(f"#examples = {ds.num_rows}.")
		assert example_idx < ds.num_rows

		example = ds[example_idx]
		print(f'{example["image"]=}.')  # PIL.Image.
		print(f'{example["ground_truth"]=}.')  # str. JSON format.

		if False:
			# Visualize.
			img = example["image"]
			gt = json.loads(example["ground_truth"])

			print("G/T:")
			print(gt)
			img.show()
			plt.show()

# REF [site] >> https://huggingface.co/ds4sd
def datasets_ds4sd_test():
	import datasets

	# DocLayNet.

	dataset_name = "ds4sd/DocLayNet"  # ~30GB.
	split = "test"  # {None, 'train', 'validation', 'test'}.
	example_idx = 0

	#ds_dict = datasets.load_dataset(dataset_name)  # datasets.DatasetDict.
	#ds = ds_dict[split]
	ds = datasets.load_dataset(dataset_name, split=split)  # datasets.Dataset.

	print(f"{ds=}.")
	"""
	DatasetDict({
		train: Dataset({
			features: ['image_id', 'image', 'width', 'height', 'doc_category', 'collection', 'doc_name', 'page_no', 'objects'],
			num_rows: 69375
		})
		validation: Dataset({
			features: ['image_id', 'image', 'width', 'height', 'doc_category', 'collection', 'doc_name', 'page_no', 'objects'],
			num_rows: 6489
		})
		test: Dataset({
			features: ['image_id', 'image', 'width', 'height', 'doc_category', 'collection', 'doc_name', 'page_no', 'objects'],
			num_rows: 4999
		})
	})
	"""

	print(f'{ds.features["image_id"]=}.')  # datasets.Value.
	print(f'{ds.features["image"]=}.')  # datasets.Image.
	print(f'{ds.features["width"]=}.')  # datasets.Value.
	print(f'{ds.features["height"]=}.')  # datasets.Value.
	print(f'{ds.features["doc_category"]=}.')  # datasets.Value.
	print(f'{ds.features["collection"]=}.')  # datasets.Value.
	print(f'{ds.features["doc_name"]=}.')  # datasets.Value.
	print(f'{ds.features["page_no"]=}.')  # datasets.Value.
	print(f'{ds.features["objects"]=}.')  # list of dicts.
	print(f"#examples = {ds.num_rows}.")
	assert example_idx < ds.num_rows

	example = ds[example_idx]
	print(f'{example["image_id"]=}.')  # int64.
	print(f'{example["image"]=}.')  # PIL.Image. 1025 x 1025.
	print(f'{example["width"]=}.')  # int32.
	print(f'{example["height"]=}.')  # int32.
	print(f'{example["doc_category"]=}.')  # str. {'financial_reports', 'scientific_articles', 'laws_and_regulations', 'government_tenders', 'manuals', 'patents'}.
	print(f'{example["collection"]=}.')  # str.
	print(f'{example["doc_name"]=}.')  # str. PDF file.
	print(f'{example["page_no"]=}.')  # int64.
	# REF [site] >> https://cocodataset.org/#format-data
	#	{
	#		'id': int64,
	#		'image_id': str,
	#		'category_id': int, 
	#			Category: {'Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'}.
	#		'bbox': [x, y, w, h], float32,
	#		'segmentation': [[x1, y1, x2, y2, ...]], float32,
	#		'area': int64,
	#		'iscrowd': bool,
	#		'precedence': int32
	#	}
	print(f'{example["objects"]=}.')  # list of dicts. {'id', 'image_id', 'category_id', 'bbox', 'segmentation', 'area', 'iscrowd', 'precedence'}.

	if False:
		# Visualize.
		img = example["image"]
		img.show()
		plt.show()

# REF [site] >> https://huggingface.co/docs/evaluate/a_quick_tour
def evaluate_quick_tour():
	# Types of evaluations:
	#	Metric:
	#		A metric is used to evaluate a model's performance and usually involves the model's predictions as well as some ground truth labels.
	#	Comparison:
	#		A comparison is used to compare two models.
	#		This can for example be done by comparing their predictions to ground truth labels and computing their agreement.
	#	Measurement:
	#		The dataset is as important as the model trained on it.
	#		With measurements one can investigate a dataset's properties.

	import evaluate

	# Load.

	# Any metric, comparison, or measurement is loaded with the evaluate.load function:
	accuracy = evaluate.load("accuracy")
	# If you want to make sure you are loading the right type of evaluation (especially if there are name clashes) you can explicitly pass the type:
	word_length = evaluate.load("word_length", module_type="measurement")

	# Community modules.
	# Besides the modules implemented in Hugging Face Evaluate you can also load any community module by specifying the repository ID of the metric implementation:
	element_count = evaluate.load("lvwerra/element_count", module_type="measurement")

	# List available modules.
	print(f"Available modules: {evaluate.list_evaluation_modules(module_type='comparison', include_community=False, with_details=True)}.")

	# Module attributes.
	accuracy = evaluate.load("accuracy")
	print(f"Accuracy description: {accuracy.description}.")
	print(f"Accuracy citation: {accuracy.citation}.")
	print(f"Accuracy features: {accuracy.features}.")

	#-----
	# Compute.
	#	All-in-one.
	#	Incremental.

	# How to compute.
	print(f"Accuracy = {accuracy.compute(references=[0, 1, 0, 1], predictions=[1, 0, 0, 1])}.")

	# Calculate a single metric or a batch of metrics.
	# If you are only creating single predictions at a time you can use add():
	for ref, pred in zip([0, 1, 0, 1], [1, 0, 0, 1]):
		accuracy.add(references=ref, predictions=pred)
	print(f"Accuracy: {accuracy.compute()}.")

	# When getting predictions and references in batches you can use add_batch() which adds a list elements for later processing:
	for refs, preds in zip([[0, 1], [0, 1]], [[1,0], [0, 1]]):
		accuracy.add_batch(references=refs, predictions=preds)
	print(f"Accuracy = {accuracy.compute()}.")

	"""
	for model_inputs, gold_standards in evaluation_dataset:
		predictions = model(model_inputs)
		metric.add_batch(references=gold_standards, predictions=predictions)
	print(f"Accuracy = {accuracy.compute()}.")
	"""

	# Distributed evaluation.

	#-----
	# Combining several evaluations.

	clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
	print(f"Metrics: {clf_metrics.compute(predictions=[0, 1, 0], references=[0, 1, 1])}.")

	#-----
	# Save and push to the Hub.
	if False:
		result = accuracy.compute(references=[0, 1, 0, 1], predictions=[1, 0, 0, 1])

		hyperparams = {"model": "bert-base-uncased"}
		evaluate.save("./results/", experiment="run 42", **result, **hyperparams)

		evaluate.push_to_hub(
			model_id="huggingface/gpt2-wikitext2",  # Model repository on hub.
			metric_value=0.5,  # Metric value.
			metric_type="bleu",  # Metric name, e.g. accuracy.name.
			metric_name="BLEU",  # Pretty name which is displayed.
			dataset_type="wikitext",  # Dataset name on the hub.
			dataset_name="WikiText",  # Pretty name.
			dataset_split="test",  # Dataset split used.
			task_type="text-generation",  # Task id, see https://github.com/huggingface/datasets/blob/master/src/datasets/utils/resources/tasks.json
			task_name="Text Generation"  # Pretty name for task.
		)

	#-----
	# Evaluator.

	import transformers
	import datasets

	pipe = transformers.pipeline("text-classification", model="lvwerra/distilbert-imdb", device=0)
	data = datasets.load_dataset("imdb", split="test").shuffle().select(range(1000))
	metric = evaluate.load("accuracy")

	task_evaluator = evaluate.evaluator("text-classification")
	results = task_evaluator.compute(
		model_or_pipeline=pipe, data=data, metric=metric,
		label_mapping={"NEGATIVE": 0, "POSITIVE": 1},
	)
	print(f"Results: {results}.")

	results = task_evaluator.compute(
		model_or_pipeline=pipe, data=data, metric=metric,
		label_mapping={"NEGATIVE": 0, "POSITIVE": 1},
		strategy="bootstrap", n_resamples=200
	)
	print(f"Results: {results}.")

	#-----
	# Visualization.

	import evaluate.visualization

	data = [
		{"accuracy": 0.99, "precision": 0.8, "f1": 0.95, "latency_in_seconds": 33.6},
		{"accuracy": 0.98, "precision": 0.87, "f1": 0.91, "latency_in_seconds": 11.2},
		{"accuracy": 0.98, "precision": 0.78, "f1": 0.88, "latency_in_seconds": 87.6},
		{"accuracy": 0.88, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 101.6}
	]
	model_names = ["Model 1", "Model 2", "Model 3", "Model 4"]
	plot = evaluate.visualization.radar_plot(data=data, model_names=model_names)
	plot.show()

	#-----
	# Running evaluation on a suite of tasks.

	class Suite(evaluate.EvaluationSuite):
		def __init__(self, name):
			super().__init__(name)

			self.suite = [
				evaluate.evaluation_suite.SubTask(
					task_type="text-classification",
					data="imdb",
					split="test[:1]",
					args_for_task={
						"metric": "accuracy",
						"input_column": "text",
						"label_column": "label",
						"label_mapping": {
							"LABEL_0": 0.0,
							"LABEL_1": 1.0
						}
					}
				),
				evaluate.evaluation_suite.SubTask(
					task_type="text-classification",
					data="sst2",
					split="test[:1]",
					args_for_task={
						"metric": "accuracy",
						"input_column": "sentence",
						"label_column": "label",
						"label_mapping": {
							"LABEL_0": 0.0,
							"LABEL_1": 1.0
						}
					}
				)
			]

	suite = evaluate.EvaluationSuite.load("mathemakitten/sentiment-evaluation-suite")
	results = suite.run("huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli")
	print(f"Results: {results}.")

# REF [site] >> https://github.com/huggingface/accelerate
def accelerate_simple_example():
	import time
	import torch
	import datasets
	import accelerate  # (+).

	#-----
	# REF [site] >> https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py
	import transformers

	tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
	dataset = datasets.load_dataset("glue", "mrpc")
	accelerator = accelerate.Accelerator()  # (+).

	batch_size = 16
	num_epochs = 10
	init_lr = 2e-5

	def tokenize_function(examples):
		# max_length=None => use the model max length (it's actually the default).
		outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
		return outputs

	tokenized_datasets = dataset.map(
		tokenize_function,
		batched=True,
		remove_columns=["idx", "sentence1", "sentence2"],
	)

	# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the transformers library.
	tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

	def collate_fn(examples):
		# On TPU it's best to pad everything to the same length or training will be very slow.
		if accelerator.distributed_type == accelerate.DistributedType.TPU:
			return tokenizer.pad(examples, padding="max_length", max_length=128, return_tensors="pt")
		return tokenizer.pad(examples, padding="longest", return_tensors="pt")

	# Instantiate dataloaders.
	dataloader = torch.utils.data.DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
	#eval_dataloader = torch.utils.data.DataLoader(tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

	# Instantiate the model (we build the model here so that the seed also control new weights initialization).
	model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)

	# We could avoid this line since the accelerator is set with 'device_placement=True' (default value).
	# Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
	# creation otherwise training will not work on TPU ('accelerate' will kindly throw an error to make us aware of that).
	#model = model.to(accelerator.device)

	# Instantiate optimizer.
	optimizer = torch.optim.AdamW(params=model.parameters(), lr=init_lr)

	# Instantiate scheduler.
	lr_scheduler = transformers.get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=100,
		num_training_steps=(len(dataloader) * num_epochs),
	)

	#--------------------
	if True:
		"""
		#device = "cpu"  # (-).
		accelerator = accelerate.Accelerator()  # (+).
		device = accelerator.device  # (+).

		model = torch.nn.Transformer().to(device)
		optimizer = torch.optim.Adam(model.parameters())

		dataset = datasets.load_dataset("my_dataset")  # Error.
		dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)
		"""
		device = accelerator.device  # (+).
		model = model.to(device)

		#model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)  # (+).
		model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)  # (+).
		#model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

		print("Training...")
		start_time = time.time()
		model.train()
		for epoch in range(num_epochs):
			"""
			for source, targets in dataloader:
				source = source.to(device)
				targets = targets.to(device)

				optimizer.zero_grad()

				output = model(source)
				loss = torch.nn.functional.cross_entropy(output, targets)

				#loss.backward()  # (-).
				accelerator.backward(loss)  # (+).

				optimizer.step()
				#lr_scheduler.step()
			"""
			for step, batch in enumerate(dataloader):
				# We could avoid this line since we set the accelerator with 'device_placement=True'.
				batch.to(device)

				optimizer.zero_grad()

				outputs = model(**batch)
				loss = outputs.loss

				#loss.backward()  # (-).
				accelerator.backward(loss)  # (+).

				optimizer.step()
				#lr_scheduler.step()
			print(f"Epoch #{epoch} done.")
		print(f"Trained: {time.time() - start_time} secs.")

	#--------------------
	if True:
		"""
		#device = "cpu"  # (-).
		accelerator = accelerate.Accelerator()  # (+).

		#model = torch.nn.Transformer().to(device)  # (-).
		model = torch.nn.Transformer()  # (+).
		optimizer = torch.optim.Adam(model.parameters())

		dataset = datasets.load_dataset("my_dataset")  # Error.
		dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)
		"""

		#model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)  # (+).
		model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)  # (+).
		#model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

		print("Training...")
		start_time = time.time()
		model.train()
		for epoch in range(num_epochs):
			"""
			for source, targets in dataloader:
				#source = source.to(device)  # (-).
				#targets = targets.to(device)  # (-).

				optimizer.zero_grad()

				output = model(source)
				loss = torch.nn.functional.cross_entropy(output, targets)

				#loss.backward()  # (-).
				accelerator.backward(loss)  # (+).

				optimizer.step()
				#lr_scheduler.step()
			"""
			for step, batch in enumerate(dataloader):
				#batch = batch.to(device)  # (-).

				optimizer.zero_grad()

				outputs = model(**batch)
				loss = outputs.loss

				#loss.backward()  # (-).
				accelerator.backward(loss)  # (+).

				optimizer.step()
				#lr_scheduler.step()
			print(f"Epoch #{epoch} done.")
		print(f"Trained: {time.time() - start_time} secs.")

# REF [site] >> https://huggingface.co/docs/accelerate/quicktour
def accelerate_quicktour():
	import torch
	import datasets
	import accelerate

	accelerator = accelerate.Accelerator()

	#-----
	# Distributed evaluation.

	dataset = datasets.load_dataset("my_dataset")  # Error.
	val_dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)

	val_dataloader = accelerator.prepare(val_dataloader)

	model.eval()
	for inputs, targets in val_dataloader:
		with torch.no_grad():
			predictions = model(inputs)

		# Gather all predictions and targets.
		all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
		# Example of use with a *Datasets.Metric*.
		#metric.add_batch(all_predictions, all_targets)

	#-----
	# Training on TPU.

	"""
	if accelerator.distributed_type == accelerate.DistributedType.TPU:
		# Do something of static shape.
	else:
		# Go crazy and be dynamic.
	"""

	#-----
	# Other caveats.

	# Execute a statement only on one processes.
	# Defer execution.
	# Saving/loading a model.
	# Saving/loading entire states.
	# Gradient clipping.

	# Mixed precision training.
	"""
	with accelerator.autocast():
		loss = complex_loss_function(outputs, target):

	if not accelerator.optimizer_step_was_skipped:
		lr_scheduler.step()
	"""

	# Gradient accumulation.
	"""
	accelerator = Accelerator(gradient_accumulation_steps=2)
	model, optimizer, training_dataloader = accelerator.prepare(model, optimizer, training_dataloader)

	for input, label in training_dataloader:
		with accelerator.accumulate(model):
			predictions = model(input)
			loss = loss_function(predictions, label)
			accelerator.backward(loss)
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()
	"""

	#-----
	# DeepSpeed.

	raise NotImplementedError

def main():
	# Hugging Face Hub.

	#hub_quickstart()
	#hub_searching_the_hub_guide()
	#hub_how_to_inference_guide()

	#hub_model_info_test()

	#--------------------
	# Datasets.
	#	https://huggingface.co/datasets

	#datasets_quick_example()
	#datasets_quicktour()
	#datasets_tutorials()

	# How-to guides.
	#	https://huggingface.co/docs/datasets/how_to
	#datasets_how_to_guides__general_use()
	#datasets_how_to_guides__audio()  # Not yet implemented.
	#datasets_how_to_guides__vision()  # Not yet implemented.
	#datasets_how_to_guides__text()  # Not yet implemented.
	#datasets_how_to_guides__tabular()  # Not yet implemented.
	#datasets_how_to_guides__dataset_repository()  # Not yet implemented.

	#-----
	# Language processing.
	#datasets_hugging_face_test()  # Wikipedia, TextVQA, WikiSQL, WikiTQ.
	#datasets_hugging_face_m4_test()  # VQAv2, VaTeX, TextCaps, NoCaps.
	#datasets_nielsr_test()  # FUNSD.
	datasets_naver_clova_test()  # CORD, SynthDoG.
	#datasets_ds4sd_test()  # DocLayNet.

	#--------------------
	# Evaluate.

	#evaluate_quick_tour()

	#--------------------
	# Accelerate.
	#	A library that enables the same PyTorch code to be run across any distributed configuration by adding just four lines of code!

	#accelerate_simple_example()
	#accelerate_quicktour()  # Not yet completed.

	# Language:
	#	https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py
	#	https://github.com/huggingface/accelerate/blob/main/examples/complete_nlp_example.py

	# Vision:
	#	https://github.com/huggingface/accelerate/blob/main/examples/cv_example.py
	#	https://github.com/huggingface/accelerate/blob/main/examples/complete_cv_example.py

	#--------------------
	# Transformers.
	#	Refer to ../language_processing/hugging_face_transformers_test.py.

	#-----
	# Object detection.
	#	Refer to ../object_detection/hugging_face_object_detection_test.py.

	#-----
	# Segmentation.
	#	Refer to ../segmentation/hugging_face_segmentation_test.py.

	#--------------------
	# Diffusers.
	#	Refer to ../machine_vision/diffusion_model_test.py.

	#--------------------
	# Parameter-Efficient Fine-Tuning (PEFT).
	#	LoRA.
	#	Prompt engineering: Prefix-Tuning, P-Tuning, Prompt Tuning.
	#
	#	Refer to ./hugging_face_peft_test.py.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
