#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/NVIDIA/Isaac-GR00T
def getting_started():
	from gr00t.data.dataset import LeRobotSingleDataset
	from gr00t.data.embodiment_tags import EmbodimentTag
	from gr00t.data.dataset import ModalityConfig
	from gr00t.experiment.data_config import DATA_CONFIG_MAP

	# 1. Data Format & Loading
	#	https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/0_load_dataset.ipynb

	# Get the data config
	data_config = DATA_CONFIG_MAP["gr1_arms_only"]

	# Get the modality configs and transforms
	modality_config = data_config.modality_config()
	transforms = data_config.transform()

	# This is a LeRobotSingleDataset object that loads the data from the given dataset path.
	dataset = LeRobotSingleDataset(
		dataset_path="demo_data/robot_sim.PickNPlace",
		modality_configs=modality_config,
		transforms=None,  # We can choose to not apply any transforms
		embodiment_tag=EmbodimentTag.GR1,  # The embodiment to use
	)

	# This is an example of how to access the data.
	dataset[5]

	#-----
	# 2. Inference
	#	https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/1_gr00t_inference.ipynb

	from gr00t.model.policy import Gr00tPolicy
	from gr00t.data.embodiment_tags import EmbodimentTag

	# 1. Load the modality config and transforms, or use above
	modality_config = ComposedModalityConfig(...)
	transforms = ComposedModalityTransform(...)

	# 2. Load the dataset
	dataset = LeRobotSingleDataset(.....<Same as above>....)

	# 3. Load pre-trained model
	policy = Gr00tPolicy(
		model_path="nvidia/GR00T-N1-2B",
		modality_config=modality_config,
		modality_transform=transforms,
		embodiment_tag=EmbodimentTag.GR1,
		device="cuda"
	)

	# 4. Run inference
	action_chunk = policy.get_action(dataset[0])

	#-----
	# 3. Fine-Tuning
	#	https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/2_finetuning.ipynb
	#	https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/3_new_embodiment_finetuning.ipynb

	# User can run the finetuning script below to finetune the model with the example dataset.
	#	A tutorial is available in getting_started/2_finetuning.ipynb.

	# Then run the finetuning script:
	#	# first run --help to see the available arguments
	#	python scripts/gr00t_finetune.py --help
	#
	#	# then run the script
	#	python scripts/gr00t_finetune.py --dataset-path ./demo_data/robot_sim.PickNPlace --num-gpus 1
	#
	#	# run using Lora Parameter Eifficient Fine-Tuning
	#	python scripts/gr00t_finetune.py  --dataset-path ./demo_data/robot_sim.PickNPlace --num-gpus 1 --lora_rank 64  --lora_alpha 128  --batch-size 32

	#-----
	# 4. Evaluation

	# Run the newly trained model:
	#	python scripts/inference_service.py --server \
	#		--model_path <MODEL_PATH> \
	#		--embodiment_tag new_embodiment \
	#		--data_config <DATA_CONFIG>

	# Run the offline evaluation script:
	#	python scripts/eval_policy.py --plot \
	#		--dataset_path <DATASET_PATH> \
	#		--embodiment_tag new_embodiment \
	#		--data_config <DATA_CONFIG>

def main():
	# References
	#	https://www.grootn1.org/
	#	https://github.com/NVIDIA/Isaac-GR00T

	# Install:
	#	git clone https://github.com/NVIDIA/Isaac-GR00T
	#	pip install -e .
	#	pip install --no-build-isolation flash-attn==2.7.1.post4 

	getting_started()  # Not yet completed

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
