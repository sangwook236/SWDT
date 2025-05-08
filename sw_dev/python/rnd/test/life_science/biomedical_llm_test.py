#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import transformers

# REF [site] >> https://huggingface.co/docs/transformers/en/model_doc/biogpt
def biogpt_example():
	if False:
		# Initializing a BioGPT microsoft/biogpt style configuration
		configuration = transformers.BioGptConfig()

		# Initializing a model from the microsoft/biogpt style configuration
		model = transformers.BioGptModel(configuration)

		# Accessing the model configuration
		configuration = model.config

	if True:
		tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/biogpt")
		model = transformers.BioGptModel.from_pretrained("microsoft/biogpt")

		inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
		outputs = model(**inputs)

		last_hidden_states = outputs.last_hidden_state

	if True:
		tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/biogpt")
		model = transformers.BioGptForCausalLM.from_pretrained("microsoft/biogpt")

		inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
		outputs = model(**inputs, labels=inputs["input_ids"])
		loss = outputs.loss
		logits = outputs.logits

	if True:
		tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/biogpt")
		model = transformers.BioGptForTokenClassification.from_pretrained("microsoft/biogpt")

		inputs = tokenizer(
			"HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
		)

		with torch.no_grad():
			logits = model(**inputs).logits

		predicted_token_class_ids = logits.argmax(-1)

		# Note that tokens are classified rather then input words which means that
		# there might be more predicted token classes than words.
		# Multiple token classes might account for the same word
		predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

		labels = predicted_token_class_ids
		loss = model(**inputs, labels=labels).loss

	if True:
		# Example of single-label classification

		tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/biogpt")
		model = transformers.BioGptForSequenceClassification.from_pretrained("microsoft/biogpt")

		inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

		with torch.no_grad():
			logits = model(**inputs).logits

		predicted_class_id = logits.argmax().item()

		# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
		num_labels = len(model.config.id2label)
		model = transformers.BioGptForSequenceClassification.from_pretrained("microsoft/biogpt", num_labels=num_labels)

		labels = torch.tensor([1])
		loss = model(**inputs, labels=labels).loss

	if True:
		# Example of multi-label classification

		tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/biogpt")
		model = transformers.BioGptForSequenceClassification.from_pretrained("microsoft/biogpt", problem_type="multi_label_classification")

		inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

		with torch.no_grad():
			logits = model(**inputs).logits

		predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

		# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
		num_labels = len(model.config.id2label)
		model = transformers.BioGptForSequenceClassification.from_pretrained(
			"microsoft/biogpt", num_labels=num_labels, problem_type="multi_label_classification"
		)

		labels = torch.sum(
			torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
		).to(torch.float)
		loss = model(**inputs, labels=labels).loss

# REF [site] >>
#	https://huggingface.co/docs/transformers/en/model_doc/ernie
#	https://huggingface.co/nghuyong
def ernie_health_example():
	# Models:
	#	nghuyong/ernie-3.0-nano-zh

	tokenizer = transformers.AutoTokenizer.from_pretrained("nghuyong/ernie-health-zh")
	model = transformers.AutoModel.from_pretrained("nghuyong/ernie-health-zh")

def main():
	#biogpt_example()  # BioGPT
	#ernie_health_example()  # ERNIE-health

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
