#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import transformers

# REF [site] >>
#	https://huggingface.co/docs/transformers/model_doc/decision_transformer
#	https://huggingface.co/edbeeching
#	https://github.com/huggingface/transformers/blob/main/examples/research_projects/decision_transformer/run_decision_transformer.py
def decision_transformer_example():
	# Models:
	#	edbeeching/decision-transformer-gym-halfcheetah-medium.
	#	edbeeching/decision-transformer-gym-halfcheetah-medium-replay.
	#	edbeeching/decision-transformer-gym-halfcheetah-expert.
	#	edbeeching/decision-transformer-gym-hopper-medium.
	#	edbeeching/decision-transformer-gym-hopper-medium-replay.
	#	edbeeching/decision-transformer-gym-hopper-expert.
	#	edbeeching/decision-transformer-gym-hopper-expert-new.
	#	edbeeching/decision-transformer-gym-walker2d-medium.
	#	edbeeching/decision-transformer-gym-walker2d-medium-replay.

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	if False:
		# Initializing a DecisionTransformer configuration.
		configuration = transformers.DecisionTransformerConfig()

		# Initializing a model (with random weights) from the configuration.
		model = transformers.DecisionTransformerModel(configuration)

		# Accessing the model configuration.
		configuration = model.config

	if True:
		import gym

		model = transformers.DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
		model = model.to(device)
		model.eval()  # Evaluation.

		env = gym.make("Hopper-v3")
		state_dim = env.observation_space.shape[0]
		act_dim = env.action_space.shape[0]

		state = env.reset()
		states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
		actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
		rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
		target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
		timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
		attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

		# Forward pass.
		with torch.no_grad():
			state_preds, action_preds, return_preds = model(
				states=states,
				actions=actions,
				rewards=rewards,
				returns_to_go=target_return,
				timesteps=timesteps,
				attention_mask=attention_mask,
				return_dict=False,
			)

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/trajectory_transformer
def trajectory_transformer_example():
	# Models:
	#	CarlCochet/trajectory-transformer-ant-medium-v2
	#	CarlCochet/trajectory-transformer-ant-medium-replay-v2
	#	CarlCochet/trajectory-transformer-ant-medium-expert-v2
	#	CarlCochet/trajectory-transformer-ant-expert-v2.
	#	CarlCochet/trajectory-transformer-halfcheetah-medium-replay-v2
	#	CarlCochet/trajectory-transformer-halfcheetah-medium-v2.
	#	CarlCochet/trajectory-transformer-halfcheetah-medium-expert-v2
	#	CarlCochet/trajectory-transformer-halfcheetah-expert-v2
	#	CarlCochet/trajectory-transformer-hopper-medium-v2
	#	CarlCochet/trajectory-transformer-hopper-medium-replay-v2
	#	CarlCochet/trajectory-transformer-hopper-medium-expert-v2
	#	CarlCochet/trajectory-transformer-hopper-expert-v2
	#	CarlCochet/trajectory-transformer-walker2d-medium-v2
	#	CarlCochet/trajectory-transformer-walker2d-medium-replay-v2
	#	CarlCochet/trajectory-transformer-walker2d-medium-expert-v2
	#	CarlCochet/trajectory-transformer-walker2d-expert-v2

	import numpy as np

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	if False:
		# Initializing a TrajectoryTransformer CarlCochet/trajectory-transformer-halfcheetah-medium-v2 style configuration
		configuration = transformers.TrajectoryTransformerConfig()

		# Initializing a model (with random weights) from the CarlCochet/trajectory-transformer-halfcheetah-medium-v2 style configuration
		model = transformers.TrajectoryTransformerModel(configuration)

		# Accessing the model configuration
		configuration = model.config

	if True:
		model = transformers.TrajectoryTransformerModel.from_pretrained("CarlCochet/trajectory-transformer-halfcheetah-medium-v2")
		model.to(device)
		model.eval()

		observations_dim, action_dim, batch_size = 17, 6, 256
		seq_length = observations_dim + action_dim + 1

		trajectories = torch.LongTensor([np.random.permutation(seq_length) for _ in range(batch_size)]).to(device)
		targets = torch.LongTensor([np.random.permutation(seq_length) for _ in range(batch_size)]).to(device)

		outputs = model(
			trajectories,
			targets=targets,
			use_cache=True,
			output_attentions=True,
			output_hidden_states=True,
			return_dict=True,
		)

# REF [site] >> https://huggingface.co/nvidia
def nemotron_4_example():
	# Models:
	#	nvidia/Nemotron-4-340B-Base
	#	nvidia/Nemotron-4-340B-Reward
	#	nvidia/Nemotron-4-340B-Instruct

	# Inference server:
	#	https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo
	#
	#	docker pull nvcr.io/nvidia/nemo:25.04

	# Usage:
	#	Deployment and inference with Nemotron-4-340B can be done in three steps using NeMo Framework:
	#	1. Create a Python script to interact with the deployed model.
	#	2. Create a Bash script to start the inference server.
	#	3. Schedule a Slurm job to distribute the model across 2 nodes and associate them with the inference server.

	raise NotImplementedError

# REF [site] >> https://huggingface.co/LiquidAI
def lfm2_example():
	# Models:
	#	LiquidAI/LFM2-350M
	#	LiquidAI/LFM2-700M
	#	LiquidAI/LFM2-1.2B
	#	LiquidAI/LFM2-350M-GGUF
	#	LiquidAI/LFM2-700M-GGUF
	#	LiquidAI/LFM2-1.2B-GGUF

	"""
	Chat template:
		<|startoftext|><|im_start|>system
		You are a helpful assistant trained by Liquid AI.<|im_end|>
		<|im_start|>user
		What is C. elegans?<|im_end|>
		<|im_start|>assistant
		It's a tiny nematode that lives in temperate soil environments.<|im_end|>

	Tool use:
		1. Function definition: LFM2 takes JSON function definitions as input (JSON objects between <|tool_list_start|> and <|tool_list_end|> special tokens), usually in the system prompt
		2. Function call: LFM2 writes Pythonic function calls (a Python list between <|tool_call_start|> and <|tool_call_end|> special tokens), as the assistant answer.
		3. Function execution: The function call is executed and the result is returned (string between <|tool_response_start|> and <|tool_response_end|> special tokens), as a "tool" role.
		4. Final answer: LFM2 interprets the outcome of the function call to address the original user prompt in plain text.

		<|startoftext|><|im_start|>system
		List of tools: <|tool_list_start|>[{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|tool_list_end|><|im_end|>
		<|im_start|>user
		What is the current status of candidate ID 12345?<|im_end|>
		<|im_start|>assistant
		<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
		<|im_start|>tool
		<|tool_response_start|>{"candidate_id": "12345", "status": "Interview Scheduled", "position": "Clinical Research Associate", "date": "2023-11-20"}<|tool_response_end|><|im_end|>
		<|im_start|>assistant
		The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the position of Clinical Research Associate, with an interview date set for 2023-11-20.<|im_end|>
	"""

	# Load model and tokenizer
	model_id = "LiquidAI/LFM2-350M"
	#model_id = "LiquidAI/LFM2-700M"
	#model_id = "LiquidAI/LFM2-1.2B"

	model = transformers.AutoModelForCausalLM.from_pretrained(
		model_id,
		device_map="auto",
		torch_dtype="bfloat16",
	#	attn_implementation="flash_attention_2" <- uncomment on compatible GPU
	)
	tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

	# Generate answer
	prompt = "What is C. elegans?"
	input_ids = tokenizer.apply_chat_template(
		[{"role": "user", "content": prompt}],
		add_generation_prompt=True,
		return_tensors="pt",
		tokenize=True,
	).to(model.device)

	output = model.generate(
		input_ids,
		do_sample=True,
		temperature=0.3,
		min_p=0.15,
		repetition_penalty=1.05,
		max_new_tokens=512,
	)

	print(tokenizer.decode(output[0], skip_special_tokens=False))

	# <|startoftext|><|im_start|>user
	# What is C. elegans?<|im_end|>
	# <|im_start|>assistant
	# C. elegans, also known as Caenorhabditis elegans, is a small, free-living
	# nematode worm (roundworm) that belongs to the phylum Nematoda.

# REF [site] >> https://huggingface.co/microsoft
def florence_example():
	# Models:
	#	microsoft/Florence-2-base
	#	microsoft/Florence-2-base-ft
	#	microsoft/Florence-2-large
	#	microsoft/Florence-2-large-ft

	import requests
	from PIL import Image

	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
	model_name = "microsoft/Florence-2-base"
	#model_name = "microsoft/Florence-2-base-ft"
	#model_name = "microsoft/Florence-2-large"
	#model_name = "microsoft/Florence-2-large-ft"

	model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
	processor = transformers.AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

	prompt = "<OD>"

	url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
	image = Image.open(requests.get(url, stream=True).raw)

	inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

	generated_ids = model.generate(
		input_ids=inputs["input_ids"],
		pixel_values=inputs["pixel_values"],
		max_new_tokens=1024,
		#max_new_tokens=4096,
		num_beams=3,
		do_sample=False,
	)
	generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

	parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

	print(parsed_answer)

# REF [site] >>
#	https://huggingface.co/nvidia
#	https://github.com/NVIDIA/Cosmos
#	https://github.com/NVIDIA/Cosmos-Tokenizer
def cosmos_example():
	# Models:
	#	nvidia/Cosmos-0.1-Tokenizer-CI8x8
	#	nvidia/Cosmos-0.1-Tokenizer-CV4x8x8
	#	nvidia/Cosmos-0.1-Tokenizer-CV8x8x8
	#	nvidia/Cosmos-0.1-Tokenizer-CV8x16x16
	#	nvidia/Cosmos-0.1-Tokenizer-DV4x8x8
	#	nvidia/Cosmos-0.1-Tokenizer-DV8x8x8
	#	nvidia/Cosmos-0.1-Tokenizer-DV8x16x16
	#
	#	nvidia/Cosmos-1.0-Diffusion-7B-Decoder-DV8x16x16ToCV8x8x8
	#	nvidia/Cosmos-1.0-Diffusion-7B-Text2World
	#	nvidia/Cosmos-1.0-Diffusion-14B-Text2World
	#	nvidia/Cosmos-1.0-Diffusion-7B-Video2World
	#	nvidia/Cosmos-1.0-Diffusion-14B-Video2World
	#	nvidia/Cosmos-1.0-Autoregressive-4B
	#	nvidia/Cosmos-1.0-Autoregressive-5B-Video2World
	#	nvidia/Cosmos-1.0-Autoregressive-12B
	#	nvidia/Cosmos-1.0-Autoregressive-13B-Video2World
	#	nvidia/Cosmos-1.0-Prompt-Upsampler-12B-Text2World
	#	nvidia/Cosmos-1.0-Guardrail
	#	nvidia/Cosmos-1.0-Tokenizer-DV8x16x16
	#	nvidia/Cosmos-1.0-Tokenizer-CV8x8x8

	# Inference:
	#	Cosmos Installation
	#		https://github.com/NVIDIA/Cosmos/blob/main/INSTALL.md
	#	Cosmos Diffusion-based World Foundation Models
	#		https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/README.md
	#	Cosmos Diffusion-based World Foundation Models: NeMo Framework User Guide
	#		https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/inference/README.md
	#	Cosmos Autoregressive-based World Foundation Models
	#		https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/README.md

	# Post-training:
	#	Cosmos Diffusion-based World Foundation Models: NeMo Framework User Guide
	#		https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md
	#	Cosmos Autoregressive-based World Foundation Models: NeMo Framework User Guide
	#		https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md

	raise NotImplementedError

def main():
	# Language modeling

	# Refer to ./language_model_test.py

	#--------------------
	# Document

	# Refer to ./document_processing_test.py

	#--------------------
	# Text

	# Refer to ./text_processing_test.py
	# Refer to ./ocr_test.py

	#--------------------
	# Natural language processing (NLP)

	# Refer to ./nlp_test.py

	#--------------------
	# Speech processing
	#	Speech recognition
	#	Speech synthesis
	#	Speech-to-speech

	# Refer to ./speech_processing_test.py

	#--------------------
	# Sequence processing

	# Refer to sequence_processing_test.py

	#--------------------
	# Biomedical

	# Refer to biomedical_llm_test.py

	#--------------------
	# Reinforcement learning

	#decision_transformer_example()  # Decision transformer
	#trajectory_transformer_example()  # Trajectory transformer

	#--------------------
	# Foundation models

	# Nemotron-4-340B is a large language model (LLM) that can be used as part of a synthetic data generation pipeline to create training data that helps researchers and developers build their own LLMs.
	#nemotron_4_example()  # Nemotron-4. Not yet implemented
	#lfm2_example()  # Liquid Foundation Models 2 (LFM2)

	#-----
	# Vision foundation models

	florence_example()  # Florence-2

	#-----
	# World models

	#cosmos_example()  # Not yet implemented

	#--------------------
	# AI agents

	# Refer to ai_agent_test.py

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
