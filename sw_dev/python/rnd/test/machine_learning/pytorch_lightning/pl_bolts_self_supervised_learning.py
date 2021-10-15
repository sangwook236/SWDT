#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Self-supervised learning.
#	REF [site] >>
#		https://lightning-bolts.readthedocs.io/en/latest/deprecated/models/self_supervised.html
#		https://lightning-bolts.readthedocs.io/en/latest/deprecated/transforms/self_supervised.html
#		https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html
#		https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_utils.html
#		https://pytorch-lightning-bolts.readthedocs.io/en/latest/transforms.html
#	AMDIM, BYOL, CPC v2, Moco v2, SimCLR, SwAV, SimSiam.

import torch, torchvision
import pytorch_lightning as pl

# REF [site] >> https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html
def simple_simclr_example():
	from pl_bolts.models.self_supervised import SimCLR
	from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform, SimCLREvalDataTransform

	# Load ResNet50 pretrained using SimCLR on ImageNet.
	weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
	simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)

	#train_dataset = MyDataset(transforms=SimCLRTrainDataTransform())
	#val_dataset = MyDataset(transforms=SimCLREvalDataTransform())
	train_dataset = torchvision.datasets.CIFAR10("", train=True, download=True, transform=SimCLRTrainDataTransform())
	val_dataset = torchvision.datasets.CIFAR10("", train=False, download=True, transform=SimCLREvalDataTransform())

	# SimCLR needs a lot of compute!
	model = SimCLR(gpus=2, num_samples=len(train_dataset), batch_size=32, dataset="cifar10")

	trainer = pl.Trainer(gpus=2, accelerator="ddp")
	trainer.fit(
		model,
		torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=12),
		torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=12),
	)

	#--------------------
	simclr_resnet50 = simclr.encoder
	simclr_resnet50.eval()

	#my_dataset = SomeDataset()
	my_dataset = val_dataset
	for batch in my_dataset:
		x, y = batch
		out = simclr_resnet50(x)

# REF [site] >> https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html
def mix_and_match_any_part_or_subclass_example():
	from pl_bolts.models.self_supervised import CPC_v2
	from pl_bolts.losses.self_supervised_learning import FeatureMapContrastiveTask
	from pl_bolts.datamodules import CIFAR10DataModule
	from pl_bolts.models.self_supervised.cpc import CPCTrainTransformsCIFAR10, CPCEvalTransformsCIFAR10
	from pytorch_lightning.plugins import DDPPlugin

	# Data module.
	dm = CIFAR10DataModule(num_workers=12, batch_size=32)
	dm.train_transforms = CPCTrainTransformsCIFAR10()
	dm.val_transforms = CPCEvalTransformsCIFAR10()

	# Model.
	amdim_task = FeatureMapContrastiveTask(comparisons="01, 11, 02", bidirectional=True)
	model = CPC_v2(encoder="cpc_encoder", contrastive_task=amdim_task)

	# Fit.
	trainer = pl.Trainer(gpus=2, accelerator="ddp", plugins=DDPPlugin(find_unused_parameters=False))
	trainer.fit(model, datamodule=dm)

# REF [site] >> https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html
def byol_example():
	from pl_bolts.models.self_supervised import BYOL
	from pl_bolts.datamodules import CIFAR10DataModule
	from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform, SimCLREvalDataTransform

	# Data module.
	dm = CIFAR10DataModule(num_workers=12, batch_size=32)
	dm.train_transforms = SimCLRTrainDataTransform(input_height=32)
	dm.val_transforms = SimCLREvalDataTransform(input_height=32)

	# Model.
	model = BYOL(num_classes=10)

	# Fit.
	trainer = pl.Trainer(gpus=2, accelerator="ddp")
	trainer.fit(model, datamodule=dm)

	#--------------------
	# CLI command:
	#	CIFAR-10:
	#		python byol_module.py --gpus 1
	#	ImageNet:
	#		python byol_module.py --gpus 8 --dataset imagenet2012 --data_dir /path/to/imagenet/ --meta_dir /path/to/folder/with/meta.bin/ --batch_size 32

# REF [site] >> https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html
def cpc_v2_example():
	from pl_bolts.models.self_supervised import CPC_v2
	from pl_bolts.datamodules import CIFAR10DataModule
	from pl_bolts.models.self_supervised.cpc import CPCTrainTransformsCIFAR10, CPCEvalTransformsCIFAR10
	from pytorch_lightning.plugins import DDPPlugin

	# Data module.
	dm = CIFAR10DataModule(num_workers=12, batch_size=32)
	dm.train_transforms = CPCTrainTransformsCIFAR10()
	dm.val_transforms = CPCEvalTransformsCIFAR10()

	# Model.
	model = CPC_v2(encoder="cpc_encoder")

	# Fit.
	trainer = pl.Trainer(gpus=2, accelerator="ddp", plugins=DDPPlugin(find_unused_parameters=False))
	trainer.fit(model, datamodule=dm)

	#--------------------
	# CIFAR-10 pretrained model:
	weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-cifar10-v4-exp3/epoch%3D474.ckpt"
	# STL-10 pretrained model:
	#weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-stl10-v0-exp3/epoch%3D624.ckpt"
	cpc_v2 = CPC_v2.load_from_checkpoint(weight_path, strict=False)

	cpc_v2.freeze()

	#--------------------
	# CLI command:
	#	Finetune:
	#		python cpc_finetuner.py --ckpt_path path/to/checkpoint.ckpt --dataset cifar10 --gpus 1

def moco_v2_example():
	from pl_bolts.models.self_supervised import Moco_v2
	from pl_bolts.datamodules import CIFAR10DataModule
	from pl_bolts.models.self_supervised.moco import Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms
	from pytorch_lightning.plugins import DDPPlugin

	# Data module.
	dm = CIFAR10DataModule(num_workers=12, batch_size=32)
	dm.train_transforms = Moco2TrainCIFAR10Transforms()
	dm.val_transforms = Moco2EvalCIFAR10Transforms()

	# Model.
	model = Moco_v2()

	# Fit.
	trainer = pl.Trainer(gpus=2, accelerator="ddp", plugins=DDPPlugin(find_unused_parameters=False))
	trainer.fit(model, datamodule=dm)

	#--------------------
	# CLI command:
	#	CIFAR-10:
	#		python moco2_module.py --gpus 1
	#	ImageNet:
	#		python moco2_module.py --gpus 8 --dataset imagenet2012 --data_dir /path/to/imagenet/ --meta_dir /path/to/folder/with/meta.bin/ --batch_size 32

# REF [site] >> https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html
def simclr_example():
	from pl_bolts.models.self_supervised import SimCLR
	from pl_bolts.datamodules import CIFAR10DataModule
	from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform, SimCLREvalDataTransform

	# Data module.
	dm = CIFAR10DataModule(num_workers=12, batch_size=32)
	dm.train_transforms = SimCLRTrainDataTransform(input_height=32)
	dm.val_transforms = SimCLREvalDataTransform(input_height=32)

	# Model.
	model = SimCLR(gpus=2, num_samples=dm.num_samples, batch_size=dm.batch_size, dataset="cifar10")

	# Fit.
	trainer = pl.Trainer(gpus=2, accelerator="ddp")
	trainer.fit(model, datamodule=dm)

	#--------------------
	# CIFAR-10 pretrained model:
	weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
	# ImageNet pretrained model:
	#weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
	simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)

	simclr.freeze()

	#--------------------
	# CLI command:
	#	CIFAR-10:
	#		Pretrain:
	#			python simclr_module.py --gpus 8 --dataset cifar10 --batch_size 256 -- num_workers 16 --optimizer sgd --learning_rate 1.5 --lars_wrapper --exclude_bn_bias --max_epochs 800 --online_ft
	#		Finetune:
	#			python simclr_finetuner.py --gpus 4 --ckpt_path path/to/simclr/ckpt --dataset cifar10 --batch_size 64 --num_workers 8 --learning_rate 0.3 --num_epochs 100
	#	ImageNet:
	#		Pretrain:
	#			python simclr_module.py --dataset imagenet --data_path path/to/imagenet
	#		Finetune:
	#			python simclr_finetuner.py --gpus 8 --ckpt_path path/to/simclr/ckpt --dataset imagenet --data_dir path/to/imagenet/dataset --batch_size 256 --num_workers 16 --learning_rate 0.8 --nesterov True --num_epochs 90

# REF [site] >> https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html
def swav_example():
	from pl_bolts.models.self_supervised import SwAV
	from pl_bolts.datamodules import STL10DataModule
	from pl_bolts.models.self_supervised.swav.transforms import SwAVTrainDataTransform, SwAVEvalDataTransform
	from pl_bolts.transforms.dataset_normalizations import stl10_normalization

	batch_size = 128

	# Data module.
	dm = STL10DataModule(data_dir=".", num_workers=16, batch_size=batch_size)
	dm.train_dataloader = dm.train_dataloader_mixed
	dm.val_dataloader = dm.val_dataloader_mixed
	dm.train_transforms = SwAVTrainDataTransform(normalize=stl10_normalization())
	dm.val_transforms = SwAVEvalDataTransform(normalize=stl10_normalization())

	# Model.
	model = SwAV(
		gpus=1,
		num_samples=dm.num_unlabeled_samples,
		dataset="stl10",
		batch_size=batch_size
	)

	# Fit.
	trainer = pl.Trainer(gpus=2, accelerator="ddp", precision=16)
	trainer.fit(model, datamodule=dm)

	#--------------------
	# ImageNet pretrained model:
	weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/bolts_swav_imagenet/swav_imagenet.ckpt"
	#weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar"
	# STL-10 pretrained model:
	#weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/checkpoints/swav_stl10.pth.tar"
	swav = SwAV.load_from_checkpoint(weight_path, strict=True)

	swav.freeze()

	#--------------------
	# CLI command:
	#	Pretrain:
	#		python swav_module.py --online_ft --gpus 1 --lars_wrapper --batch_size 128 --learning_rate 1e-3 --gaussian_blur --queue_length 0 --jitter_strength 1. --nmb_prototypes 512
	#	Finetune:
	#		python swav_finetuner.py --gpus 8 --ckpt_path path/to/simclr/ckpt --dataset imagenet --data_dir path/to/imagenet/dataset --batch_size 256 --num_workers 16 --learning_rate 0.8 --nesterov True --num_epochs 90

def simsiam_example():
	from pl_bolts.models.self_supervised import SimSiam
	from pl_bolts.datamodules import CIFAR10DataModule
	from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform, SimCLREvalDataTransform

	# Data module.
	dm = CIFAR10DataModule(num_workers=12, batch_size=32)
	dm.train_transforms = SimCLRTrainDataTransform(input_height=32)
	dm.val_transforms = SimCLREvalDataTransform(input_height=32)

	# Model.
	model = SimSiam(gpus=2, num_samples=dm.num_samples, batch_size=dm.batch_size, dataset="cifar10")

	# Fit.
	trainer = pl.Trainer(gpus=2, accelerator="ddp")
	trainer.fit(model, datamodule=dm)

	#--------------------
	# CLI command:
	#	CIFAR-10:
	#		python simsiam_module.py --gpus 1
	#	ImageNet:
	#		python simsiam_module.py --gpus 8 --dataset imagenet2012 --data_dir /path/to/imagenet/ --meta_dir /path/to/folder/with/meta.bin/ --batch_size 32

# REF [site] >> https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/13-contrastive-learning.html
def self_supervised_contrastive_learning_with_simclr_tutorial():
	raise NotImplementedError

def main():
	#simple_simclr_example()
	#mix_and_match_any_part_or_subclass_example()

	#byol_example()
	#cpc_v2_example()
	#moco_v2_example()
	simclr_example()
	#swav_example()
	#simsiam_example()

	#self_supervised_contrastive_learning_with_simclr_tutorial()  # Not yet implemented.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
