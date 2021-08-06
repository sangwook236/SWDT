#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import torch
import matplotlib.pyplot as plt

# REF [site] >> https://pytorch.org/tutorials/advanced/cpp_frontend.html
def inspect_generated_images():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--sample-file", required=True)
	parser.add_argument("-o", "--out-file", default="out.png")
	parser.add_argument("-d", "--dimension", type=int, default=3)
	options = parser.parse_args()

	#--------------------
	module = torch.jit.load(options.sample_file)
	images = list(module.parameters())[0]

	for idx in range(options.dimension * options.dimension):
		if idx >= images.shape[0]: break

		image = images[idx].detach().cpu().reshape(28, 28).mul(255).to(torch.uint8)
		array = image.numpy()
		axis = plt.subplot(options.dimension, options.dimension, idx + 1)
		plt.imshow(array, cmap="gray")
		axis.get_xaxis().set_visible(False)
		axis.get_yaxis().set_visible(False)

	plt.savefig(options.out_file)
	print("Saved ", options.out_file)

def main():
	inspect_generated_images()

#--------------------------------------------------------------------

# Usage:
#	python inspect_dcgan_generated_images.py -i dcgan-sample-100.pt

if '__main__' == __name__:
	main()
