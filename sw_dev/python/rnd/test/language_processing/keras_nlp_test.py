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

	# Load a BERT model.
	classifier = keras_nlp.models.BertClassifier.from_preset(
		"bert_base_en_uncased",
		num_classes=2,
	)

	# Fine-tune on IMDb movie reviews.
	classifier.fit(imdb_train, validation_data=imdb_test)

	# Predict two new examples.
	classifier.predict(["What an amazing movie!", "A total waste of my time."])

# REF [site] >> https://www.kaggle.com/models/google/gemma
def gemma_example():
	import numpy as np
	#import keras
	import keras_nlp

	if True:
		# Use generate() to do text generation.

		gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_7b_en")
		gemma_lm.generate("Keras is a", max_length=30)

		# Generate with batched prompts.
		gemma_lm.generate(["Keras is a", "I want to say"], max_length=30)

	if True:
		# Compile the generate() function with a custom sampler.

		gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_7b_en")
		gemma_lm.compile(sampler="top_k")
		gemma_lm.generate("I want to say", max_length=30)

		gemma_lm.compile(sampler=keras_nlp.samplers.BeamSampler(num_beams=2))
		gemma_lm.generate("I want to say", max_length=30)

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
		gemma_lm.generate(prompt)

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

def main():
	quickstart()

	gemma_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
