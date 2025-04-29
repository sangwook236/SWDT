#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import transformers

# REF [site] >>
#	https://huggingface.co/docs/transformers/model_doc/time_series_transformer
#	https://huggingface.co/blog/time-series-transformers
def time_series_transformer_example():
	if False:
		# Initializing a Time Series Transformer configuration with 12 time steps for prediction
		configuration = transformers.TimeSeriesTransformerConfig(prediction_length=12)

		# Randomly initializing a model (with random weights) from the configuration
		model = transformers.TimeSeriesTransformerModel(configuration)

		# Accessing the model configuration
		configuration = model.config

	if False:
		import huggingface_hub

		file = huggingface_hub.hf_hub_download(repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset")
		batch = torch.load(file)

		model = transformers.TimeSeriesTransformerModel.from_pretrained("huggingface/time-series-transformer-tourism-monthly")

		# During training, one provides both past and future values as well as possible additional features
		outputs = model(
			past_values=batch["past_values"],
			past_time_features=batch["past_time_features"],
			past_observed_mask=batch["past_observed_mask"],
			static_categorical_features=batch["static_categorical_features"],
			static_real_features=batch["static_real_features"],
			future_values=batch["future_values"],
			future_time_features=batch["future_time_features"],
		)

		last_hidden_state = outputs.last_hidden_state
		print(f"{last_hidden_state.shape=}.")

	if True:
		import huggingface_hub

		file = huggingface_hub.hf_hub_download(repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset")
		batch = torch.load(file)

		model = transformers.TimeSeriesTransformerForPrediction.from_pretrained("huggingface/time-series-transformer-tourism-monthly")

		# During training, one provides both past and future values as well as possible additional features
		outputs = model(
			past_values=batch["past_values"],
			past_time_features=batch["past_time_features"],
			past_observed_mask=batch["past_observed_mask"],
			static_categorical_features=batch["static_categorical_features"],
			static_real_features=batch["static_real_features"],
			future_values=batch["future_values"],
			future_time_features=batch["future_time_features"],
		)

		loss = outputs.loss
		loss.backward()

		# During inference, one only provides past values as well as possible additional features
		# and the model autoregressively generates future values
		outputs = model.generate(
			past_values=batch["past_values"],
			past_time_features=batch["past_time_features"],
			past_observed_mask=batch["past_observed_mask"],
			static_categorical_features=batch["static_categorical_features"],
			static_real_features=batch["static_real_features"],
			future_time_features=batch["future_time_features"],
		)

		mean_prediction = outputs.sequences.mean(dim=1)
		std_prediction = outputs.sequences.std(dim=1)  # ?

		#print(f"{model.config=}.")
		print(f"{model.config.loss=}.")  # Negative log likelihood (NLL).
		print(f"{model.config.distribution_output=}.")  # Student's t-distribution
		print(f"{model.config.prediction_length=}.")
		print(f"{outputs.sequences.shape=}.")  # (batch size, number of samples, prediction length)
		print(f"{mean_prediction.shape=}.")
		print(f"{std_prediction.shape=}.")

		#-----
		# REF [site] >> https://huggingface.co/blog/time-series-transformers

		import numpy as np
		import evaluate

		forecasts = outputs.sequences
		forecast_median = np.median(forecasts, 1)

		file = huggingface_hub.hf_hub_download(repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset")
		#file = huggingface_hub.hf_hub_download(repo_id="hf-internal-testing/tourism-monthly-batch", filename="val-batch.pt", repo_type="dataset")
		batch = torch.load(file)
		assert len(forecast_median) == len(batch["future_values"]) == len(batch["past_values"])
		batch_size = len(forecast_median)

		#-----
		mase_metric = evaluate.load("evaluate-metric/mase")
		mape_metric = evaluate.load("evaluate-metric/mape")
		smape_metric = evaluate.load("evaluate-metric/smape")

		mase_metrics, mape_metrics, smape_metrics = [], [], []
		for idx in range(batch_size):
			mase = mase_metric.compute(
				predictions=forecast_median[idx],
				references=batch["future_values"][idx].numpy(),
				training=batch["past_values"][idx].numpy(),
				#periodicity=gluonts.time_feature.get_seasonality(freq),
			)
			mase_metrics.append(mase["mase"])

			mape = mape_metric.compute(
				predictions=forecast_median[idx],
				references=batch["future_values"][idx].numpy(),
			)
			mape_metrics.append(mape["mape"])

			smape = smape_metric.compute(
				predictions=forecast_median[idx],
				references=batch["future_values"][idx].numpy(),
			)
			smape_metrics.append(smape["smape"])
		print(f"MASE = {np.mean(mase_metrics)}, MAPE = {np.mean(mape_metrics)}, sMAPE = {np.mean(smape_metrics)}.")

		#-----
		mase_metric = evaluate.load("evaluate-metric/mase", "multilist")
		mape_metric = evaluate.load("evaluate-metric/mape", "multilist")
		smape_metric = evaluate.load("evaluate-metric/smape", "multilist")

		mase = mase_metric.compute(
			predictions=forecast_median.transpose(),  # (timesteps, #sequences).
			references=batch["future_values"].numpy().transpose(),  # (timesteps, #sequences).
			training=batch["past_values"].numpy().transpose(),  # (timesteps, #sequences).
			multioutput="raw_values",
			#periodicity=gluonts.time_feature.get_seasonality(freq),
		)
		mape = mape_metric.compute(
			predictions=forecast_median.transpose(),  # (timesteps, #sequences).
			references=batch["future_values"].numpy().transpose(),  # (timesteps, #sequences).
			multioutput="raw_values",
		)
		smape = smape_metric.compute(
			predictions=forecast_median.transpose(),  # (timesteps, #sequences).
			references=batch["future_values"].numpy().transpose(),  # (timesteps, #sequences).
			multioutput="raw_values",
		)
		print(f'MASE = {np.mean(mase["mase"])}, MAPE = {np.mean(mape["mape"])}, sMAPE = {np.mean(smape["smape"])}.')

# REF [site] >>
#	https://huggingface.co/docs/transformers/en/model_doc/mamba
#	https://huggingface.co/state-spaces
#	https://github.com/state-spaces/mamba
def mamba_example():
	# Models:
	#	state-spaces/mamba-130m
	#	state-spaces/mamba-130m-hf
	#	state-spaces/mamba-370m
	#	state-spaces/mamba-370m-hf
	#	state-spaces/mamba-390m-hf
	#	state-spaces/mamba-790m
	#	state-spaces/mamba-790m-hf
	#	state-spaces/mamba-1.4b
	#	state-spaces/mamba-1.4b-hf
	#	state-spaces/mamba-2.8b
	#	state-spaces/mamba-2.8b-slimpj
	#	state-spaces/mamba-2.8b-hf

	if True:
		# A simple generation example

		tokenizer = transformers.AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
		model = transformers.MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
		input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

		out = model.generate(input_ids, max_new_tokens=10)
		print(tokenizer.batch_decode(out))

	if True:
		# Peft finetuning
		#	The slow version is not very stable for training, and the fast one needs float32!

		from datasets import load_dataset
		from trl import SFTTrainer
		from peft import LoraConfig

		model_id = "state-spaces/mamba-130m-hf"
		tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
		model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
		dataset = load_dataset("Abirate/english_quotes", split="train")
		training_args = transformers.TrainingArguments(
			output_dir="./results",
			num_train_epochs=3,
			per_device_train_batch_size=4,
			logging_dir="./logs",
			logging_steps=10,
			learning_rate=2e-3,
		)
		lora_config = LoraConfig(
			r=8,
			target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
			task_type="CAUSAL_LM",
			bias="none",
		)
		trainer = SFTTrainer(
			model=model,
			tokenizer=tokenizer,
			args=training_args,
			peft_config=lora_config,
			train_dataset=dataset,
			dataset_text_field="quote",
		)
		trainer.train()

	if False:
		# Initializing a Mamba configuration
		configuration = transformers.MambaConfig()

		# Initializing a model (with random weights) from the configuration
		model = transformers.MambaModel(configuration)

		# Accessing the model configuration
		configuration = model.config

	if False:
		tokenizer = transformers.AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
		model = transformers.MambaModel.from_pretrained("state-spaces/mamba-130m-hf")

		inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
		outputs = model(**inputs)

		last_hidden_states = outputs.last_hidden_state

	if True:
		tokenizer = transformers.AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
		model = transformers.MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

		inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
		outputs = model(**inputs, labels=inputs["input_ids"])
		loss = outputs.loss
		logits = outputs.logits

# REF [site] >>
#	https://huggingface.co/docs/transformers/en/model_doc/mamba2
#	https://huggingface.co/state-spaces
#	https://github.com/state-spaces/mamba
def mamba2_example():
	# Models:
	#	state-spaces/mamba2-130m
	#	state-spaces/mamba2-370m
	#	state-spaces/mamba2-780m
	#	state-spaces/mamba2-1.3b
	#	state-spaces/mamba2-2.7b
	#	state-spaces/mamba2attn-2.7b

	if True:
		# A simple generation example

		model_id = "mistralai/Mamba-Codestral-7B-v0.1"
		tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision="refs/pr/9", from_slow=True, legacy=False)
		model = transformers.MambaForCausalLM.from_pretrained(model_id, revision="refs/pr/9")
		input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

		out = model.generate(input_ids, max_new_tokens=10)
		print(tokenizer.batch_decode(out))

	if True:
		# A draft script for finetuning

		from datasets import load_dataset
		from trl import SFTTrainer
		from peft import LoraConfig

		model_id = "mistralai/Mamba-Codestral-7B-v0.1"
		tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision="refs/pr/9", from_slow=True, legacy=False)
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.padding_side = "left" #enforce padding side left

		model = transformers.Mamba2ForCausalLM.from_pretrained(model_id, revision="refs/pr/9")
		dataset = load_dataset("Abirate/english_quotes", split="train")
		# Without CUDA kernels, batch size of 2 occupies one 80GB device
		# but precision can be reduced.
		# Experiments and trials welcome!
		training_args = transformers.TrainingArguments(
			output_dir="./results",
			num_train_epochs=3,
			per_device_train_batch_size=2,
			logging_dir="./logs",
			logging_steps=10,
			learning_rate=2e-3,
		)
		lora_config = LoraConfig(
			r=8,
			target_modules=["embeddings", "in_proj", "out_proj"],
			task_type="CAUSAL_LM",
			bias="none",
		)
		trainer = SFTTrainer(
			model=model,
			tokenizer=tokenizer,
			args=training_args,
			peft_config=lora_config,
			train_dataset=dataset,
			dataset_text_field="quote",
		)
		trainer.train()

	if False:
		# Initializing a Mamba2 configuration
		configuration = transformers.Mamba2Config()

		# Initializing a model (with random weights) from the configuration
		model = transformers.Mamba2Model(configuration)

		# Accessing the model configuration
		configuration = model.config

	if False:
		tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/mamba-codestral-7B-v0.1")
		model = transformers.Mamba2Model.from_pretrained("mistralai/mamba-codestral-7B-v0.1")

		inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
		outputs = model(**inputs)

		last_hidden_states = outputs.last_hidden_state

	if True:
		tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/mamba-codestral-7B-v0.1")
		model = transformers.Mamba2ForCausalLM.from_pretrained("mistralai/mamba-codestral-7B-v0.1")

		inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
		outputs = model(**inputs, labels=inputs["input_ids"])
		loss = outputs.loss
		logits = outputs.logits

def main():
	#time_series_transformer_example()  # Probabilistic time series transformer.

	#mamba_example()  # Mamba.
	mamba2_example()  # Mamba-2.

#--------------------------------------------------------------------

if "_main__" == __name__:
	main()
