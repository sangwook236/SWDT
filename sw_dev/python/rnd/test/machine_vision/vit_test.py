#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torchvision

# REF [site] >> https://github.com/lucidrains/vit-pytorch
def vit_pytorch_test():
	from vit_pytorch import ViT

	v = ViT(
		image_size=256,
		patch_size=32,
		num_classes=1000,
		dim=1024,
		depth=6,
		heads=16,
		mlp_dim=2048,
		dropout=0.1,
		emb_dropout=0.1
	)

	img = torch.randn(1, 3, 256, 256)

	preds = v(img)  # (1, 1000).
	print(f"(ViT) Predictions: shape = {preds.shape}.")

	#-----
	from vit_pytorch import SimpleViT

	v = SimpleViT(
		image_size=256,
		patch_size=32,
		num_classes=1000,
		dim=1024,
		depth=6,
		heads=16,
		mlp_dim=2048
	)

	img = torch.randn(1, 3, 256, 256)

	preds = v(img)  # (1, 1000).
	print(f"(SimpleViT) Predictions: shape = {preds.shape}.")

	#-----
	# "Training data-efficient image transformers & distillation through attention", arXiv 2021 (DeiT).
	from vit_pytorch.distill import DistillableViT, DistillWrapper

	teacher = torchvision.models.resnet50(pretrained=True)

	v = DistillableViT(
		image_size=256,
		patch_size=32,
		num_classes=1000,
		dim=1024,
		depth=6,
		heads=8,
		mlp_dim=2048,
		dropout=0.1,
		emb_dropout=0.1
	)

	distiller = DistillWrapper(
		student=v,
		teacher=teacher,
		temperature=3,  # Temperature of distillation.
		alpha=0.5,  # Trade between main loss and distillation loss.
		hard=False  # Whether to use soft or hard distillation.
	)

	img = torch.randn(2, 3, 256, 256)
	labels = torch.randint(0, 1000, (2,))

	loss = distiller(img, labels)
	loss.backward()

	# After lots of training above ...
	preds = v(img)  # (2, 1000).
	print(f"(DeiT) Predictions: shape = {preds.shape}.")

	#-----
	# "DeepViT: Towards Deeper Vision Transformer", arXiv 2021.
	from vit_pytorch.deepvit import DeepViT

	v = DeepViT(
		image_size=256,
		patch_size=32,
		num_classes=1000,
		dim=1024,
		depth=6,
		heads=16,
		mlp_dim=2048,
		dropout=0.1,
		emb_dropout=0.1
	)

	img = torch.randn(1, 3, 256, 256)

	preds = v(img)  # (1, 1000).
	print(f"(DeepViT) Predictions: shape = {preds.shape}.")

	#-----
	# "Going deeper with Image Transformers", arXiv 2021 (CaiT).
	from vit_pytorch.cait import CaiT

	v = CaiT(
		image_size=256,
		patch_size=32,
		num_classes=1000,
		dim=1024,
		depth=12,  # Depth of transformer for patch to patch attention only.
		cls_depth=2,  # Depth of cross attention of CLS tokens to patch.
		heads=16,
		mlp_dim=2048,
		dropout=0.1,
		emb_dropout=0.1,
		layer_dropout=0.05  # Randomly dropout 5% of the layers.
	)

	img = torch.randn(1, 3, 256, 256)

	preds = v(img)  # (1, 1000).
	print(f"(CaiT) Predictions: shape = {preds.shape}.")

	#-----
	# "MaxViT: Multi-Axis Vision Transformer", arXiv 2022.
	from vit_pytorch.max_vit import MaxViT

	v = MaxViT(
		num_classes=1000,
		dim_conv_stem=64,  # Dimension of the convolutional stem, would default to dimension of first layer if not specified.
		dim=96,  # Dimension of first layer, doubles every layer.
		dim_head=32,  # Dimension of attention heads, kept at 32 in paper.
		depth=(2, 2, 5, 2),  # Number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention.
		window_size=7,  # Window size for block and grids.
		mbconv_expansion_rate=4,  # Expansion rate of MBConv.
		mbconv_shrinkage_rate=0.25,  # Shrinkage rate of squeeze-excitation in MBConv.
		dropout=0.1  # Dropout.
	)

	img = torch.randn(2, 3, 224, 224)

	preds = v(img)  # (2, 1000).
	print(f"(MaxViT) Predictions: shape = {preds.shape}.")

	#-----
	# "Masked Autoencoders Are Scalable Vision Learners", arXiv 2021 (MAE).
	from vit_pytorch import ViT, MAE

	v = ViT(
		image_size=256,
		patch_size=32,
		num_classes=1000,
		dim=1024,
		depth=6,
		heads=8,
		mlp_dim=2048
	)

	mae = MAE(
		encoder=v,
		masking_ratio=0.75,  # The paper recommended 75% masked patches.
		decoder_dim=512,  # Paper showed good results with just 512.
		decoder_depth=6  # Anywhere from 1 to 8.
	)

	images = torch.randn(8, 3, 256, 256)

	loss = mae(images)
	loss.backward()

	# That's all!
	# Do the above in a for loop many times with a lot of images and your vision transformer will learn.

	# Save your improved vision transformer.
	#torch.save(v.state_dict(), "./trained-vit.pt")

	#-----
	# "Emerging Properties in Self-Supervised Vision Transformers", arXiv 2021 (DINO).
	from vit_pytorch import ViT, Dino

	model = ViT(
		image_size=256,
		patch_size=32,
		num_classes=1000,
		dim=1024,
		depth=6,
		heads=8,
		mlp_dim=2048
	)

	learner = Dino(
		model,
		image_size=256,
		hidden_layer="to_latent",  # Hidden layer name or index, from which to extract the embedding.
		projection_hidden_size=256,  # Projector network hidden dimension.
		projection_layers=4,  # Number of layers in projection network.
		num_classes_K=65336,  # Output logits dimensions (referenced as K in paper).
		student_temp=0.9,  # Student temperature.
		teacher_temp=0.04,  # Teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs.
		local_upper_crop_scale=0.4,  # Upper bound for local crop - 0.4 was recommended in the paper.
		global_lower_crop_scale=0.5,  # Lower bound for global crop - 0.5 was recommended in the paper.
		moving_average_decay=0.9,  # Moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok.
		center_moving_average_decay=0.9,  # Moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok.
	)

	opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

	def sample_unlabelled_images():
		return torch.randn(20, 3, 256, 256)

	for _ in range(100):
		images = sample_unlabelled_images()
		loss = learner(images)
		opt.zero_grad()
		loss.backward()
		opt.step()
		learner.update_moving_average()  # Update moving average of teacher encoder and teacher centers.

	# Save your improved network.
	#torch.save(model.state_dict(), "./pretrained-net.pt")

def main():
	# Model			Layers		Hidden size		MLP size	Heads	Params
	#----------------------------------------------------------------------
	# ViT-Base		12			768				3072		12		86M
	# ViT-Large		24			1024			4096		16		307M
	# ViT-Huge		32			1280			5120		16		632M

	# REF [file] >>
	#	${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch/pytorch_transformer.py
	#	${SWDT_PYTHON_HOME}/rnd/test/language_processing/hugging_face_transformers_test.py

	vit_pytorch_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
