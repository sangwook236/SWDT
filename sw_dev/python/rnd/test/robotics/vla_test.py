#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://huggingface.co/physical-intelligence
def fast_example():
	# Models:
	#	physical-intelligence/fast

	# Install:
	#	pip install transformers scipy

	import numpy as np
	import transformers

	if True:
		# Using the Universal Action Tokenizer

		# Load the tokenizer from the Hugging Face hub
		tokenizer = transformers.AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

		# Tokenize & decode action chunks (we use dummy data here)
		action_data = np.random.rand(256, 50, 14)  # one batch of action chunks
		tokens = tokenizer(action_data)  # tokens = list[int]
		decoded_actions = tokenizer.decode(tokens)

	if False:
		# Training a new Action Tokenizer on Your Own Data

		# First, we download the tokenizer from the Hugging Face model hub
		# Here, we will not use the pre-trained tokenizer weights, but only the source code
		# to train a new tokenizer on our own data.
		tokenizer = transformers.AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

		# Load your action data for tokenizer training
		# Chunks do not need to be of the same length, we will use dummy data
		action_data = np.random.rand(4000, 50, 14)

		# Train the new tokenizer, depending on your dataset size this can take a few minutes
		tokenizer = tokenizer.fit(action_data)

		# Save the new tokenizer, optionally push it to the Hugging Face model hub
		tokenizer.save_pretrained("<your_local_path>")
		tokenizer.push_to_hub("YourUsername/my_new_tokenizer")

# REF [site] >> https://huggingface.co/openvla
def openvla_example():
	# Models:
	#	openvla/openvla-v01-7b
	#	openvla/openvla-7b

	from PIL import Image
	import torch
	import transformers

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	if False:
		# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...):
		#	pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt

		# Load Processor & VLA
		processor = transformers.AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
		vla = transformers.AutoModelForVision2Seq.from_pretrained(
			"openvla/openvla-v01-7b",
			attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
			torch_dtype=torch.bfloat16, 
			low_cpu_mem_usage=True, 
			trust_remote_code=True
		).to(device)

		# Grab image input & format prompt (note inclusion of system prompt due to Vicuña base model)
		image: Image.Image = get_from_camera(...)
		system_prompt = (
			"A chat between a curious user and an artificial intelligence assistant. "
			"The assistant gives helpful, detailed, and polite answers to the user's questions."
		)
		prompt = f"{system_prompt} USER: What action should the robot take to {<INSTRUCTION>}? ASSISTANT:"

		# Predict Action (7-DoF; un-normalize for BridgeV2)
		inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
		action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

		# Execute...
		robot.act(action, ...)

	if True:
		# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...):
		#	pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt

		# Load Processor & VLA
		processor = transformers.AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
		vla = transformers.AutoModelForVision2Seq.from_pretrained(
			"openvla/openvla-7b",
			attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
			torch_dtype=torch.bfloat16, 
			low_cpu_mem_usage=True, 
			trust_remote_code=True
		).to(device)

		# Grab image input & format prompt
		image: Image.Image = get_from_camera(...)
		prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

		# Predict Action (7-DoF; un-normalize for BridgeV2)
		inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
		action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

		# Execute...
		robot.act(action, ...)

# REF [site] >>
#	https://www.physicalintelligence.company/blog/pi0
#	https://www.physicalintelligence.company/research/fast
#	https://github.com/Physical-Intelligence/openpi
def openpi_test():
	# Install:
	#	git clone https://github.com/Physical-Intelligence/openpi.git
	#	git submodule update --init --recursive
	#	GIT_LFS_SKIP_SMUDGE=1 pip install -e .
	#		NOTE: GIT_LFS_SKIP_SMUDGE=1 is needed to pull LeRobot as a dependency.

	from openpi.training import config
	from openpi.policies import policy_config
	from openpi.shared import download

	config = config.get_config("pi0_fast_droid")
	checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")

	# Create a trained policy.
	policy = policy_config.create_trained_policy(config, checkpoint_dir)

	# Run inference on a dummy example.
	'''
	example = {
		"observation/exterior_image_1_left": ...,
		"observation/wrist_image_left": ...,
		...
		"prompt": "pick up the fork"
	}
	'''
	action_chunk = policy.infer(example)["actions"]

# REF [site] >>
#	https://huggingface.co/blog/pi0
#	https://github.com/huggingface/lerobot/tree/main/lerobot/common/policies/pi0
#	https://github.com/huggingface/lerobot/tree/main/lerobot/common/policies/pi0fast
def lerobot_pi0_test():
	# Models:
	#	lerobot/pi0
	#	lerobot/pi0fast_base

	# Install:
	#	pip install -e ".[pi0]"

	if False:
		# Not working

		policy = Pi0Policy.from_pretrained("lerobot/pi0")  # pi0
		#policy = PI0FASTPolicy.from_pretrained("lerobot/pi0fast_base")  # pi0 + FAST

		action = policy.select_action(batch)

# REF [site] >> https://github.com/lucidrains/pi-zero-pytorch
def pi_zero_pytorch_test():
	raise NotImplementedError

def main():
	fast_example()  # FAST
	#openvla_example()  # OpenVLA. Not yet completed

	# pi0
	#openpi_test()  # Not yet completed
	#lerobot_pi0_test()  # Not yet completed
	#pi_zero_pytorch_test()  # Not yet implemented

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
