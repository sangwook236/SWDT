#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import torch
import transformers
import datasets
import peft
from tqdm import tqdm

# REF [site] >> https://github.com/huggingface/peft
def peft_get_started():
	model_name_or_path = "bigscience/mt0-large"
	tokenizer_name_or_path = "bigscience/mt0-large"

	peft_config = peft.LoraConfig(task_type=peft.TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
	#peft_config = peft.get_peft_config(config_dict=...)

	model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
	model = peft.get_peft_model(model, peft_config)

	print("Trainable parameters:")
	model.print_trainable_parameters()  # Output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282

# REF [site] >> https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq.ipynb
def peft_lora_seq2seq_example():
	os.environ["TOKENIZERS_PARALLELISM"] = "false"

	device = "cuda"
	model_name_or_path = "bigscience/mt0-large"
	tokenizer_name_or_path = "bigscience/mt0-large"

	checkpoint_name = "financial_sentiment_analysis_lora_v1.pt"
	text_column = "sentence"
	label_column = "text_label"
	max_length = 128
	lr = 1e-3
	num_epochs = 3
	batch_size = 8

	# Creating model.
	peft_config = peft.LoraConfig(
		task_type=peft.TaskType.SEQ_2_SEQ_LM,
		inference_mode=False,
		r=8, lora_alpha=32, lora_dropout=0.1,
	)

	model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
	model = peft.get_peft_model(model, peft_config)

	print("Trainable parameters:")
	model.print_trainable_parameters()
	print("Model:")
	print(model)

	# Loading dataset.
	dataset = datasets.load_dataset("financial_phrasebank", "sentences_allagree")
	dataset = dataset["train"].train_test_split(test_size=0.1)
	dataset["validation"] = dataset["test"]
	del dataset["test"]

	classes = dataset["train"].features["label"].names
	dataset = dataset.map(
		lambda x: {"text_label": [classes[label] for label in x["label"]]},
		batched=True,
		num_proc=1,
	)

	print(f'{dataset["train"][0]=}.')

	# Data preprocessing.
	tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

	def preprocess_function(examples):
		inputs = examples[text_column]
		targets = examples[label_column]
		model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
		labels = tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
		labels = labels["input_ids"]
		labels[labels == tokenizer.pad_token_id] = -100
		model_inputs["labels"] = labels
		return model_inputs

	processed_datasets = dataset.map(
		preprocess_function,
		batched=True,
		num_proc=1,
		remove_columns=dataset["train"].column_names,
		load_from_cache_file=False,
		desc="Running tokenizer on dataset",
	)

	train_dataset = processed_datasets["train"]
	eval_dataset = processed_datasets["validation"]

	train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=transformers.default_data_collator, batch_size=batch_size, pin_memory=True)
	eval_dataloader = torch.utils.data.DataLoader(eval_dataset, collate_fn=transformers.default_data_collator, batch_size=batch_size, pin_memory=True)

	#-----
	# Optimizer and LR scheduler.
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	lr_scheduler = transformers.get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=0,
		num_training_steps=(len(train_dataloader) * num_epochs),
	)

	# Training and evaluation.
	model = model.to(device)

	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		for step, batch in enumerate(tqdm(train_dataloader)):
			batch = {k: v.to(device) for k, v in batch.items()}
			outputs = model(**batch)
			loss = outputs.loss
			total_loss += loss.detach().float()
			loss.backward()
			optimizer.step()
			lr_scheduler.step()
			optimizer.zero_grad()

		model.eval()
		eval_loss = 0
		eval_preds = []
		for step, batch in enumerate(tqdm(eval_dataloader)):
			batch = {k: v.to(device) for k, v in batch.items()}
			with torch.no_grad():
				outputs = model(**batch)
			loss = outputs.loss
			eval_loss += loss.detach().float()
			eval_preds.extend(tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True))

		eval_epoch_loss = eval_loss / len(eval_dataloader)
		eval_ppl = torch.exp(eval_epoch_loss)
		train_epoch_loss = total_loss / len(train_dataloader)
		train_ppl = torch.exp(train_epoch_loss)
		print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

	# Print accuracy.
	correct = 0
	total = 0
	for pred, true in zip(eval_preds, dataset["validation"]["text_label"]):
		if pred.strip() == true.strip():
			correct += 1
		total += 1
	accuracy = correct / total * 100
	print(f"{accuracy=} % on the evaluation dataset")
	print(f"{eval_preds[:10]=}")
	print(f"{dataset['validation']['text_label'][:10]=}")

	# Saving model.
	peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
	model.save_pretrained(peft_model_id)
	print(f"Model saved to {peft_model_id}.")

	#-----
	#ckpt = f"{peft_model_id}/adapter_model.bin"
	peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

	config = peft.PeftConfig.from_pretrained(peft_model_id)
	model = transformers.AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
	model = peft.PeftModel.from_pretrained(model, peft_model_id)

	model.eval()
	i = 13
	inputs = tokenizer(dataset["validation"][text_column][i], return_tensors="pt")
	print(dataset["validation"][text_column][i])
	print(inputs)

	with torch.no_grad():
		outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
		print(outputs)
		print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

# REF [site] >> https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_prefix_tuning_seq2seq.ipynb
def peft_prefix_tuning_seq2seq_example():
	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	os.environ["CUDA_VISIBLE_DEVICES"] = "3"

	device = "cuda"
	model_name_or_path = "t5-large"
	tokenizer_name_or_path = "t5-large"

	checkpoint_name = "financial_sentiment_analysis_prefix_tuning_v1.pt"
	text_column = "sentence"
	label_column = "text_label"
	max_length = 128
	lr = 1e-2
	num_epochs = 5
	batch_size = 8

	# Creating model.
	peft_config = peft.PrefixTuningConfig(
		task_type=peft.TaskType.SEQ_2_SEQ_LM,
		inference_mode=False,
		num_virtual_tokens=20,
	)

	model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
	model = peft.get_peft_model(model, peft_config)

	print("Trainable parameters:")
	model.print_trainable_parameters()
	print("Model:")
	print(model)

	# Loading dataset.
	dataset = datasets.load_dataset("financial_phrasebank", "sentences_allagree")
	dataset = dataset["train"].train_test_split(test_size=0.1)
	dataset["validation"] = dataset["test"]
	del dataset["test"]

	classes = dataset["train"].features["label"].names
	dataset = dataset.map(
		lambda x: {"text_label": [classes[label] for label in x["label"]]},
		batched=True,
		num_proc=1,
	)

	print(f'{dataset["train"][0]=}.')

	# Data preprocessing.
	tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

	def preprocess_function(examples):
		inputs = examples[text_column]
		targets = examples[label_column]
		model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
		labels = tokenizer(targets, max_length=2, padding="max_length", truncation=True, return_tensors="pt")
		labels = labels["input_ids"]
		labels[labels == tokenizer.pad_token_id] = -100
		model_inputs["labels"] = labels
		return model_inputs

	processed_datasets = dataset.map(
		preprocess_function,
		batched=True,
		num_proc=1,
		remove_columns=dataset["train"].column_names,
		load_from_cache_file=False,
		desc="Running tokenizer on dataset",
	)

	train_dataset = processed_datasets["train"]
	eval_dataset = processed_datasets["validation"]

	train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=transformers.default_data_collator, batch_size=batch_size, pin_memory=True)
	eval_dataloader = torch.utils.data.DataLoader(eval_dataset, collate_fn=transformers.default_data_collator, batch_size=batch_size, pin_memory=True)

	#-----
	# Optimizer and LR scheduler.
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	lr_scheduler = transformers.get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=0,
		num_training_steps=(len(train_dataloader) * num_epochs),
	)

	# Training and evaluation.
	model = model.to(device)

	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		for step, batch in enumerate(tqdm(train_dataloader)):
			batch = {k: v.to(device) for k, v in batch.items()}
			outputs = model(**batch)
			loss = outputs.loss
			total_loss += loss.detach().float()
			loss.backward()
			optimizer.step()
			lr_scheduler.step()
			optimizer.zero_grad()

		model.eval()
		eval_loss = 0
		eval_preds = []
		for step, batch in enumerate(tqdm(eval_dataloader)):
			batch = {k: v.to(device) for k, v in batch.items()}
			with torch.no_grad():
				outputs = model(**batch)
			loss = outputs.loss
			eval_loss += loss.detach().float()
			eval_preds.extend(tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True))

		eval_epoch_loss = eval_loss / len(eval_dataloader)
		eval_ppl = torch.exp(eval_epoch_loss)
		train_epoch_loss = total_loss / len(train_dataloader)
		train_ppl = torch.exp(train_epoch_loss)
		print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

	# Print accuracy.
	correct = 0
	total = 0
	for pred, true in zip(eval_preds, dataset["validation"]["text_label"]):
		if pred.strip() == true.strip():
			correct += 1
		total += 1
	accuracy = correct / total * 100
	print(f"{accuracy=} % on the evaluation dataset")
	print(f"{eval_preds[:10]=}")
	print(f"{dataset['validation']['text_label'][:10]=}")

	# Saving model.
	peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
	model.save_pretrained(peft_model_id)
	print(f"Model saved to {peft_model_id}.")

	#-----
	#ckpt = f"{peft_model_id}/adapter_model.bin"
	peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

	config = peft.PeftConfig.from_pretrained(peft_model_id)
	model = transformers.AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
	model = peft.PeftModel.from_pretrained(model, peft_model_id)

	model.eval()
	i = 107
	inputs = tokenizer(dataset["validation"][text_column][i], return_tensors="pt")
	print(dataset["validation"][text_column][i])
	print(inputs)

	with torch.no_grad():
		outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
		print(outputs)
		print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

# REF [site] >> https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_prefix_tuning_clm.ipynb
def peft_prefix_tuning_clm_example():
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Device: {device}.")

	model_name_or_path = "bigscience/bloomz-560m"
	tokenizer_name_or_path = "bigscience/bloomz-560m"
	peft_config = peft.PrefixTuningConfig(
		task_type=peft.TaskType.CAUSAL_LM,
		num_virtual_tokens=30,
	)

	dataset_name = "twitter_complaints"
	checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace("/", "_")
	text_column = "Tweet text"
	label_column = "text_label"
	max_length = 64
	lr = 3e-2
	num_epochs = 50
	batch_size = 8

	dataset = datasets.load_dataset("ought/raft", dataset_name)

	classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
	print(f"Classes: {classes}.")
	dataset = dataset.map(
		lambda x: {"text_label": [classes[label] for label in x["Label"]]},
		batched=True,
		num_proc=1,
	)
	print("Dataset:")
	print(dataset)
	print(f'{dataset["train"][0]=}.')

	# Data preprocessing.
	tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id
	target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
	print(target_max_length)

	def preprocess_function(examples):
		batch_size = len(examples[text_column])
		inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
		targets = [str(x) for x in examples[label_column]]
		model_inputs = tokenizer(inputs)
		labels = tokenizer(targets)
		for i in range(batch_size):
			sample_input_ids = model_inputs["input_ids"][i]
			label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
			#print(i, sample_input_ids, label_input_ids)
			model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
			labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
			model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
		#print(model_inputs)
		for i in range(batch_size):
			sample_input_ids = model_inputs["input_ids"][i]
			label_input_ids = labels["input_ids"][i]
			model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (max_length - len(sample_input_ids)) + sample_input_ids
			model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
			labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
			model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
			model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
			labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
		model_inputs["labels"] = labels["input_ids"]
		return model_inputs

	processed_datasets = dataset.map(
		preprocess_function,
		batched=True,
		num_proc=1,
		remove_columns=dataset["train"].column_names,
		load_from_cache_file=False,
		desc="Running tokenizer on dataset",
	)

	train_dataset = processed_datasets["train"]
	eval_dataset = processed_datasets["train"]

	train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=transformers.default_data_collator, batch_size=batch_size, pin_memory=True)
	eval_dataloader = torch.utils.data.DataLoader(eval_dataset, collate_fn=transformers.default_data_collator, batch_size=batch_size, pin_memory=True)

	def test_preprocess_function(examples):
		batch_size = len(examples[text_column])
		inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
		model_inputs = tokenizer(inputs)
		#print(model_inputs)
		for i in range(batch_size):
			sample_input_ids = model_inputs["input_ids"][i]
			model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (max_length - len(sample_input_ids)) + sample_input_ids
			model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
			model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
			model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
		return model_inputs

	test_dataset = dataset["test"].map(
		test_preprocess_function,
		batched=True,
		num_proc=1,
		remove_columns=dataset["train"].column_names,
		load_from_cache_file=False,
		desc="Running tokenizer on dataset",
	)

	test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=transformers.default_data_collator, batch_size=batch_size, pin_memory=True)

	print("Train batch:")
	print(next(iter(train_dataloader)))
	print("Validation batch:")
	print(next(iter(eval_dataloader)))
	print("Test batch:")
	print(next(iter(test_dataloader)))
	print(f"#train steps per epoch = {len(train_dataloader)}, #val steps per epoch = {len(eval_dataloader)}, #test steps per epoch = {len(test_dataloader)}.")

	# Creating model.
	model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
	model = peft.get_peft_model(model, peft_config)

	print("Trainable parameters:")
	model.print_trainable_parameters()
	print("Model:")
	print(model)
	print("Model PEFT config:")
	print(model.peft_config)

	#-----
	# Optimizer and LR scheduler.
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	lr_scheduler = transformers.get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=0,
		num_training_steps=(len(train_dataloader) * num_epochs),
	)

	# Training and evaluation.
	model = model.to(device)

	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		for step, batch in enumerate(tqdm(train_dataloader)):
			batch = {k: v.to(device) for k, v in batch.items()}
			#print(batch)
			#print(batch["input_ids"].shape)
			outputs = model(**batch)
			loss = outputs.loss
			total_loss += loss.detach().float()
			loss.backward()
			optimizer.step()
			lr_scheduler.step()
			optimizer.zero_grad()

		model.eval()
		eval_loss = 0
		eval_preds = []
		for step, batch in enumerate(tqdm(eval_dataloader)):
			batch = {k: v.to(device) for k, v in batch.items()}
			with torch.no_grad():
				outputs = model(**batch)
			loss = outputs.loss
			eval_loss += loss.detach().float()
			eval_preds.extend(tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True))

		eval_epoch_loss = eval_loss / len(eval_dataloader)
		eval_ppl = torch.exp(eval_epoch_loss)
		train_epoch_loss = total_loss / len(train_dataloader)
		train_ppl = torch.exp(train_epoch_loss)
		print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

	model.eval()
	i = 16
	inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', return_tensors="pt")
	print(dataset["test"][i]["Tweet text"])
	print(inputs)

	with torch.no_grad():
		inputs = {k: v.to(device) for k, v in inputs.items()}
		outputs = model.generate(
			input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
			max_new_tokens=10, eos_token_id=3,
		)
		print(outputs)
		print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

	# Saving model.
	peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
	model.save_pretrained(peft_model_id)
	print(f"Model saved to {peft_model_id}.")

	#-----
	#ckpt = f"{peft_model_id}/adapter_model.bin"
	peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

	config = peft.PeftConfig.from_pretrained(peft_model_id)
	model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
	model = peft.PeftModel.from_pretrained(model, peft_model_id)

	model.to(device)
	model.eval()
	i = 4
	inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', return_tensors="pt")
	print(dataset["test"][i]["Tweet text"])
	print(inputs)

	with torch.no_grad():
		inputs = {k: v.to(device) for k, v in inputs.items()}
		outputs = model.generate(
			input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
			max_new_tokens=10, eos_token_id=3,
		)
		print(outputs)
		print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

# REF [site] >> https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_prompt_tuning_clm.ipynb
def peft_prompt_tuning_clm_example():
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Device: {device}.")

	model_name_or_path = "bigscience/bloomz-560m"
	tokenizer_name_or_path = "bigscience/bloomz-560m"
	peft_config = peft.PromptTuningConfig(
		task_type=peft.TaskType.CAUSAL_LM,
		prompt_tuning_init=peft.PromptTuningInit.TEXT,
		num_virtual_tokens=8,
		prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
		tokenizer_name_or_path=model_name_or_path,
	)

	dataset_name = "twitter_complaints"
	checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace("/", "_")
	text_column = "Tweet text"
	label_column = "text_label"
	max_length = 64
	lr = 3e-2
	num_epochs = 50
	batch_size = 8

	dataset = datasets.load_dataset("ought/raft", dataset_name)

	classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
	print(f"Classes: {classes}.")
	dataset = dataset.map(
		lambda x: {"text_label": [classes[label] for label in x["Label"]]},
		batched=True,
		num_proc=1,
	)
	print("Dataset:")
	print(dataset)
	print(f'{dataset["train"][0]=}.')

	# Data preprocessing.
	tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id
	target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
	print(target_max_length)

	def preprocess_function(examples):
		batch_size = len(examples[text_column])
		inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
		targets = [str(x) for x in examples[label_column]]
		model_inputs = tokenizer(inputs)
		labels = tokenizer(targets)
		for i in range(batch_size):
			sample_input_ids = model_inputs["input_ids"][i]
			label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
			#print(i, sample_input_ids, label_input_ids)
			model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
			labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
			model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
		#print(model_inputs)
		for i in range(batch_size):
			sample_input_ids = model_inputs["input_ids"][i]
			label_input_ids = labels["input_ids"][i]
			model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (max_length - len(sample_input_ids)) + sample_input_ids
			model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
			labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
			model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
			model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
			labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
		model_inputs["labels"] = labels["input_ids"]
		return model_inputs

	processed_datasets = dataset.map(
		preprocess_function,
		batched=True,
		num_proc=1,
		remove_columns=dataset["train"].column_names,
		load_from_cache_file=False,
		desc="Running tokenizer on dataset",
	)

	train_dataset = processed_datasets["train"]
	eval_dataset = processed_datasets["train"]

	train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=transformers.default_data_collator, batch_size=batch_size, pin_memory=True)
	eval_dataloader = torch.utils.data.DataLoader(eval_dataset, collate_fn=transformers.default_data_collator, batch_size=batch_size, pin_memory=True)

	def test_preprocess_function(examples):
		batch_size = len(examples[text_column])
		inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
		model_inputs = tokenizer(inputs)
		#print(model_inputs)
		for i in range(batch_size):
			sample_input_ids = model_inputs["input_ids"][i]
			model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (max_length - len(sample_input_ids)) + sample_input_ids
			model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
			model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
			model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
		return model_inputs

	test_dataset = dataset["test"].map(
		test_preprocess_function,
		batched=True,
		num_proc=1,
		remove_columns=dataset["train"].column_names,
		load_from_cache_file=False,
		desc="Running tokenizer on dataset",
	)

	test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=transformers.default_data_collator, batch_size=batch_size, pin_memory=True)

	print("Train batch:")
	print(next(iter(train_dataloader)))
	print("Validation batch:")
	print(next(iter(eval_dataloader)))
	print("Test batch:")
	print(next(iter(test_dataloader)))
	print(f"#train steps per epoch = {len(train_dataloader)}, #val steps per epoch = {len(eval_dataloader)}, #test steps per epoch = {len(test_dataloader)}.")

	# Creating model.
	model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
	model = peft.get_peft_model(model, peft_config)

	print("Trainable parameters:")
	model.print_trainable_parameters()
	print("Model:")
	print(model)
	print("Model PEFT config:")
	print(model.peft_config)

	#-----
	# Optimizer and LR scheduler.
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	lr_scheduler = transformers.get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=0,
		num_training_steps=(len(train_dataloader) * num_epochs),
	)

	# Training and evaluation.
	model = model.to(device)

	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		for step, batch in enumerate(tqdm(train_dataloader)):
			batch = {k: v.to(device) for k, v in batch.items()}
			#print(batch)
			#print(batch["input_ids"].shape)
			outputs = model(**batch)
			loss = outputs.loss
			total_loss += loss.detach().float()
			loss.backward()
			optimizer.step()
			lr_scheduler.step()
			optimizer.zero_grad()

		model.eval()
		eval_loss = 0
		eval_preds = []
		for step, batch in enumerate(tqdm(eval_dataloader)):
			batch = {k: v.to(device) for k, v in batch.items()}
			with torch.no_grad():
				outputs = model(**batch)
			loss = outputs.loss
			eval_loss += loss.detach().float()
			eval_preds.extend(tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True))

		eval_epoch_loss = eval_loss / len(eval_dataloader)
		eval_ppl = torch.exp(eval_epoch_loss)
		train_epoch_loss = total_loss / len(train_dataloader)
		train_ppl = torch.exp(train_epoch_loss)
		print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

	model.eval()
	i = 33
	inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', return_tensors="pt")
	print(dataset["test"][i]["Tweet text"])
	print(inputs)

	with torch.no_grad():
		inputs = {k: v.to(device) for k, v in inputs.items()}
		outputs = model.generate(
			input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
			max_new_tokens=10, eos_token_id=3,
		)
		print(outputs)
		print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

	# Saving model.
	peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
	model.save_pretrained(peft_model_id)
	print(f"Model saved to {peft_model_id}.")

	#-----
	#ckpt = f"{peft_model_id}/adapter_model.bin"
	peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

	config = peft.PeftConfig.from_pretrained(peft_model_id)
	model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
	model = peft.PeftModel.from_pretrained(model, peft_model_id)

	model.to(device)
	model.eval()
	i = 4
	inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', return_tensors="pt")
	print(dataset["test"][i]["Tweet text"])
	print(inputs)

	with torch.no_grad():
		inputs = {k: v.to(device) for k, v in inputs.items()}
		outputs = model.generate(
			input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
			max_new_tokens=10, eos_token_id=3,
		)
		print(outputs)
		print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

def main():
	# Parameter-Efficient Fine-Tuning (PEFT).
	#	LoRA.
	#	Prompt engineering: Prefix-Tuning, P-Tuning, Prompt Tuning.

	#peft_get_started()

	peft_lora_seq2seq_example()
	#peft_prefix_tuning_seq2seq_example()
	#peft_prefix_tuning_clm_example()  # Causal language modeling.
	#peft_prompt_tuning_clm_example()  # Causal language modeling.

	# REF [file] >>
	#	${competition_HOME}/ai_connect_competiton_nipa/2023/korean_text_summarization/kogpt_fine_tuning_lora.py
	#	${competition_HOME}/ai_connect_competiton_nipa/2023/korean_text_summarization/kogpt_fine_tuning_prompt_tuning.py

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
