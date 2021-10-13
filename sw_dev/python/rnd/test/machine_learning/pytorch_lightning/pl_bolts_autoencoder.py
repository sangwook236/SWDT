#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch, torchvision
import pytorch_lightning as pl

# REF [site] >> https://pytorch-lightning-bolts.readthedocs.io/en/latest/autoencoders.html
def basic_ae_example():
	from pl_bolts.models.autoencoders import AE
	from pytorch_lightning.plugins import DDPPlugin

	# Data.
	train_dataset = torchvision.datasets.CIFAR10("", train=True, download=True, transform=torchvision.transforms.ToTensor())
	val_dataset = torchvision.datasets.CIFAR10("", train=False, download=True, transform=torchvision.transforms.ToTensor())

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=12)
	#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=12, persistent_workers=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=12)
	#val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=12, persistent_workers=True)

	# Model.
	model = AE(input_height=32)
	"""
	# Override any part of this AE to build your own variation.
	class MyAEFlavor(AE):
		def init_encoder(self, hidden_dim, latent_dim, input_width, input_height):
			encoder = YourSuperFancyEncoder(...)
			return encoder

	model = MyAEFlavor(...)
	"""

	# Fit.
	trainer = pl.Trainer(gpus=2, accelerator="ddp", plugins=DDPPlugin(find_unused_parameters=False))
	trainer.fit(model, train_loader, val_loader)

	#--------------------
	# CIFAR-10 pretrained model:
	ae = AE(input_height=32)
	print(AE.pretrained_weights_available())
	ae = ae.from_pretrained("cifar10-resnet18")

	ae.freeze()

# REF [site] >> https://pytorch-lightning-bolts.readthedocs.io/en/latest/autoencoders.html
def simple_vae_example():
	from pl_bolts.models.autoencoders import VAE
	from pytorch_lightning.plugins import DDPPlugin

	# Data.
	train_dataset = torchvision.datasets.CIFAR10("", train=True, download=True, transform=torchvision.transforms.ToTensor())
	val_dataset = torchvision.datasets.CIFAR10("", train=False, download=True, transform=torchvision.transforms.ToTensor())

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=12)
	#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=12, persistent_workers=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=12)
	#val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=12, persistent_workers=True)

	# Model.
	model = VAE(input_height=32)
	"""
	# Override any part of this AE to build your own variation.
	class MyVAEFlavor(VAE):
		def get_posterior(self, mu, std):
			# Do something other than the default.
			P = self.get_distribution(self.prior, loc=torch.zeros_like(mu), scale=torch.ones_like(std))
			return P

	model = MyVAEFlavor(...)
	"""

	# Fit.
	trainer = pl.Trainer(gpus=2, accelerator="ddp", plugins=DDPPlugin(find_unused_parameters=False))
	trainer.fit(model, train_loader, val_loader)

	#--------------------
	# CIFAR-10 pretrained model:
	vae = VAE(input_height=32)
	print(VAE.pretrained_weights_available())
	vae = vae.from_pretrained("cifar10-resnet18")

	vae.freeze()

# REF [site] >> https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/08-deep-autoencoders.html
def deep_autoencoder_tutorial():
	raise NotImplementedError

def main():
	#basic_ae_example()
	simple_vae_example()

	#deep_autoencoder_tutorial()  # Not yet implemented.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
