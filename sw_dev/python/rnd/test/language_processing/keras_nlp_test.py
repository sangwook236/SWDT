#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://keras.io/keras_nlp/
def quickstart():
	import os
	os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

	import keras_nlp
	import tensorflow_datasets as tfds

	imdb_train, imdb_test = tfds.load(
		"imdb_reviews",
		split=["train", "test"],
		as_supervised=True,
		batch_size=16,
	)

	# Load a BERT model
	classifier = keras_nlp.models.BertClassifier.from_preset(
		"bert_base_en_uncased",
		num_classes=2,
	)

	# Fine-tune on IMDb movie reviews
	classifier.fit(imdb_train, validation_data=imdb_test)

	# Predict two new examples
	predicted = classifier.predict(["What an amazing movie!", "A total waste of my time."])
	print(predicted)

# REF [site] >> https://www.kaggle.com/models/google/gemma
def gemma_example():
	import numpy as np
	#import keras
	import keras_nlp

	if True:
		# Use generate() to do text generation.

		gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_7b_en")
		generated = gemma_lm.generate("Keras is a", max_length=30)
		print(generated)

		# Generate with batched prompts.
		generated = gemma_lm.generate(["Keras is a", "I want to say"], max_length=30)
		print(generated)

	if True:
		# Compile the generate() function with a custom sampler.

		gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_7b_en")
		gemma_lm.compile(sampler="top_k")
		generated = gemma_lm.generate("I want to say", max_length=30)
		print(generated)

		gemma_lm.compile(sampler=keras_nlp.samplers.BeamSampler(num_beams=2))
		generated = gemma_lm.generate("I want to say", max_length=30)
		print(generated)

	if True:
		# Use generate() without preprocessing.

		prompt = {
			# `2, 214064, 603` maps to the start token followed by "Keras is".
			"token_ids": np.array([[2, 214064, 603, 0, 0, 0, 0]] * 2),
			# Use `"padding_mask"` to indicate values that should not be overridden.
			"padding_mask": np.array([[1, 1, 1, 0, 0, 0, 0]] * 2),
		}

		gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(
			"gemma_instruct_7b_en",
			preprocessor=None,
		)
		generated = gemma_lm.generate(prompt)
		print(generated)

	if True:
		# Call fit() on a single batch.

		features = ["The quick brown fox jumped.", "I forgot my homework."]
		gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_7b_en")
		gemma_lm.fit(x=features, batch_size=2)

	if True:
		# Call fit() without preprocessing.

		x = {
			"token_ids": np.array([[2, 214064, 603, 5271, 6044, 9581, 3, 0]] * 2),
			"padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 0]] * 2),
		}
		y = np.array([[214064, 603, 5271, 6044, 9581, 3, 0, 0]] * 2)
		sw = np.array([[1, 1, 1, 1, 1, 1, 0, 0]] * 2)

		gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(
			"gemma_instruct_7b_en",
			preprocessor=None,
		)
		gemma_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)

# REF [site] >> https://www.kaggle.com/code/nilaychauhan/get-started-with-gemma-using-kerasnlp
def get_started_with_gemma_example():
	import os
	#import keras
	import keras_nlp

	os.environ["KERAS_BACKEND"] = "jax"  # Or "tensorflow" or "torch"

	# Create a model
	gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

	print(gemma_lm.summary())

	# Generate text
	generated = gemma_lm.generate("What is the meaning of life?", max_length=64)
	print(generated)
	generated = gemma_lm.generate("How does the brain work?", max_length=64)
	print(generated)
	generated = gemma_lm.generate([
			"What is the meaning of life?",
			"How does the brain work?",
		],
		max_length=64
	)
	print(generated)

	# Try a different sampler
	gemma_lm.compile(sampler="top_k")
	generated = gemma_lm.generate("What is the meaning of life?", max_length=64)
	print(generated)

# REF [site] >> https://www.kaggle.com/code/nilaychauhan/keras-gemma-distributed-finetuning-and-inference
def gemma_distributed_finetuning_and_inference_example():
	import os
	import keras
	import keras_nlp
	import tensorflow_datasets as tfds
	import jax

	print(jax.devices())

	# The Keras 3 distribution API is only implemented for the JAX backend for now
	os.environ["KERAS_BACKEND"] = "jax"
	# Pre-allocate 90% of TPU memory to minimize memory fragmentation and allocation overhead
	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

	#-----
	# Load model

	# Uncomment the line below if you want to enable mixed precision training on GPUs
	#keras.mixed_precision.set_global_policy("mixed_bfloat16")

	# Create a device mesh with (1, 8) shape so that the weights are sharded across all 8 TPUs
	device_mesh = keras.distribution.DeviceMesh(
		(1, 8),
		["batch", "model"],
		devices=keras.distribution.list_devices(),
	)

	model_dim = "model"

	layout_map = keras.distribution.LayoutMap(device_mesh)

	# Weights that match 'token_embedding/embeddings' will be sharded on 8 TPUs
	layout_map["token_embedding/embeddings"] = (None, model_dim)
	# Regex to match against the query, key and value matrices in the decoder attention layers
	layout_map["decoder_block.*attention.*(query|key|value).*kernel"] = (None, model_dim, None)

	layout_map["decoder_block.*attention_output.*kernel"] = (None, None, model_dim)
	layout_map["decoder_block.*ffw_gating.*kernel"] = (model_dim, None)
	layout_map["decoder_block.*ffw_linear.*kernel"] = (None, model_dim)

	model_parallel = keras.distribution.ModelParallel(device_mesh, layout_map, batch_dim_name="batch")

	keras.distribution.set_distribution(model_parallel)
	gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_7b_en")

	# Verify that the model has been partitioned correctly. Let's take decoder_block_1 as an example
	decoder_block_1 = gemma_lm.backbone.get_layer("decoder_block_1")
	print(type(decoder_block_1))
	for variable in decoder_block_1.weights:
		print(f"{variable.path:<58}  {str(variable.shape):<16}  {str(variable.value.sharding.spec)}")

	# Inference before finetuning
	gemma_lm.generate("Best comedy movies in the 90s ", max_length=64)

	#-----
	# Finetune with IMDB
	imdb_train = tfds.load(
		"imdb_reviews",
		split="train",
		as_supervised=True,
		batch_size=2,
	)
	# Drop labels
	imdb_train = imdb_train.map(lambda x, y: x)

	imdb_train.unbatch().take(1).get_single_element().numpy()

	# Use a subset of the dataset for faster training.
	imdb_train = imdb_train.take(2000)

	# Enable LoRA for the model and set the LoRA rank to 4
	gemma_lm.backbone.enable_lora(rank=4)

	#-----
	# Fine-tune on the IMDb movie reviews dataset

	# Limit the input sequence length to 128 to control memory usage
	gemma_lm.preprocessor.sequence_length = 128
	# Use AdamW (a common optimizer for transformer models)
	optimizer = keras.optimizers.AdamW(
		learning_rate=5e-5,
		weight_decay=0.01,
	)
	# Exclude layernorm and bias terms from decay
	optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

	gemma_lm.compile(
		loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		optimizer=optimizer,
		weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
	)
	print(gemma_lm.summary())

	gemma_lm.fit(imdb_train, epochs=1)

	# Inference after finetuning
	generated = gemma_lm.generate("Best comedy movies in the 90s ", max_length=64)
	print(generated)

# REF [site] >> https://www.kaggle.com/code/nilaychauhan/fine-tune-gemma-models-in-keras-using-lora
def fine_tune_gemma_models_using_lora_example():
	import os
	import keras
	import keras_nlp
	import json

	os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow"
	# Avoid memory fragmentation on JAX backend
	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

	#-----
	# Load dataset
	# Format the entire example as a single string
	template = "Instruction:\n{instruction}\n\nResponse:\n{response}"

	data = []
	with open("/kaggle/input/databricks-dolly-15k/databricks-dolly-15k.jsonl") as file:
		for line in file:
			features = json.loads(line)
			# Filter out examples with context, to keep it simple
			if features["context"]:
				continue
			data.append(template.format(**features))

	# Only use 1000 training examples, to keep it fast
	data = data[:1000]

	#-----
	# Load model
	gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
	print(gemma_lm.summary())

	# Inference before fine tuning
	
	# Europe trip prompt
	prompt = template.format(
		instruction="What should I do on a trip to Europe?",
		response="",
	)
	generated = gemma_lm.generate(prompt, max_length=256)
	print(generated)

	# ELI5 photosynthesis prompt
	prompt = template.format(
		instruction="Explain the process of photosynthesis in a way that a child could understand.",
		response="",
	)
	generated = gemma_lm.generate(prompt, max_length=256)
	print(generated)

	#-----
	# LoRA fine-tuning

	# Enable LoRA for the model and set the LoRA rank to 4
	gemma_lm.backbone.enable_lora(rank=4)
	print(gemma_lm.summary())

	# Note that enabling LoRA reduces the number of trainable parameters significantly (from 2.5 billion to 1.3 million)
	# Limit the input sequence length to 512 (to control memory usage)
	gemma_lm.preprocessor.sequence_length = 512
	# Use AdamW (a common optimizer for transformer models)
	optimizer = keras.optimizers.AdamW(
		learning_rate=5e-5,
		weight_decay=0.01,
	)
	# Exclude layernorm and bias terms from decay
	optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

	gemma_lm.compile(
		loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		optimizer=optimizer,
		weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
	)
	gemma_lm.fit(data, epochs=1, batch_size=1)

	# Inference after fine-tuning

	# Europe trip prompt
	prompt = template.format(
		instruction="What should I do on a trip to Europe?",
		response="",
	)
	generated = gemma_lm.generate(prompt, max_length=256)
	print(generated)

	# ELI5 photosynthesis prompt
	prompt = template.format(
		instruction="Explain the process of photosynthesis in a way that a child could understand.",
		response="",
	)
	generated = gemma_lm.generate(prompt, max_length=256)
	print(generated)

def main():
	quickstart()

	# Gemma
	#gemma_example()

	#get_started_with_gemma_example()
	#gemma_distributed_finetuning_and_inference_example()
	#fine_tune_gemma_models_using_lora_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
