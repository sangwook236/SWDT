#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/utkuozbulak/pytorch-cnn-visualizations

import sys
sys.path.append('./pytorch_cnn_visualizations')

import os, time
import numpy as np
import torch
import torchvision
import pytorch_cnn_visualizations as cnnvis
import pytorch_cnn_visualizations.misc_functions as cnnvis_mf
import PIL.Image

# REF [file] >> ${pytorch-cnn-visualizations_HOME}/vanilla_backprop.py, guided_backprop.py, gradcam.py, guided_gradcam.py, integrated_gradients.py
def gradient_visualization_example():
	# Pick one of the examples.
	img_path, target_class = './snake.jpg', 56
	#img_path, target_class = './cat_dog.png', 243
	#img_path, target_class = './spider.png', 72

	file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]

	# Read image.
	orig_img = PIL.Image.open(img_path).convert('RGB')
	# Process image.
	img = cnnvis_mf.preprocess_image(orig_img)

	# Define model.
	pretrained_model = torchvision.models.alexnet(pretrained=True)

	#--------------------
	# Vanilla BackProp.

	import pytorch_cnn_visualizations.vanilla_backprop
	import pytorch_cnn_visualizations.smooth_grad

	start_time = time.time()
	VBP = cnnvis.vanilla_backprop.VanillaBackprop(pretrained_model)

	# Generate gradients.
	vanilla_grads = VBP.generate_gradients(img, target_class)
	# Save colored gradients.
	cnnvis_mf.save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')

	# Convert to grayscale.
	grayscale_vanilla_grads = cnnvis_mf.convert_to_grayscale(vanilla_grads)
	# Save grayscale gradients.
	cnnvis_mf.save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')

	print('Vanilla BackProp completed: {} secs.'.format(time.time() - start_time))

	# Gradient x Image.
	start_time = time.time()
	grad_times_image = vanilla_grads[0] * img.detach().numpy()[0]
	# Convert to grayscale.
	grayscale_vanilla_grads = cnnvis_mf.convert_to_grayscale(grad_times_image)
	# Save grayscale gradients.
	cnnvis_mf.save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_times_image_gray')

	print('Vanilla BackProp times image completed: {} secs.'.format(time.time() - start_time))

	# Smooth Gradient.
	start_time = time.time()
	param_n = 50  # Number of images (n) to average over.
	param_sigma_multiplier = 4
	smooth_grad = cnnvis.smooth_grad.generate_smooth_grad(VBP, img, target_class, param_n, param_sigma_multiplier)
	# Save colored gradients
	cnnvis_mf.save_gradient_images(smooth_grad, file_name_to_export + '_Vanilla_BP_SmoothGrad_color')

	# Convert to grayscale.
	grayscale_smooth_grad = cnnvis_mf.convert_to_grayscale(smooth_grad)
	# Save grayscale gradients.
	cnnvis_mf.save_gradient_images(grayscale_smooth_grad, file_name_to_export + '_Vanilla_BP_SmoothGrad_gray')

	print('Vanilla BackProp Smooth Grad completed: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Guided BackProp.

	import pytorch_cnn_visualizations.guided_backprop

	start_time = time.time()
	GBP = cnnvis.guided_backprop.GuidedBackprop(pretrained_model)

	# Get gradients.
	guided_grads = GBP.generate_gradients(img, target_class)
	# Save colored gradients.
	cnnvis_mf.save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')

	# Convert to grayscale.
	grayscale_guided_grads = cnnvis_mf.convert_to_grayscale(guided_grads)
	# Save grayscale gradients.
	cnnvis_mf.save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')

	# Positive and negative saliency maps.
	pos_sal, neg_sal = cnnvis_mf.get_positive_negative_saliency(guided_grads)
	cnnvis_mf.save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
	cnnvis_mf.save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')

	print('Guided BackProp completed: {} secs.'.format(time.time() - start_time))

	# Gradient x Image.
	start_time = time.time()
	grad_times_image = guided_grads[0] * img.detach().numpy()[0]
	# Convert to grayscale.
	grayscale_guided_grads = cnnvis_mf.convert_to_grayscale(grad_times_image)
	# Save grayscale gradients.
	cnnvis_mf.save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_times_image_gray')

	print('Guided BackProp times image completed: {} secs.'.format(time.time() - start_time))

	# Smooth Gradient.
	start_time = time.time()
	param_n = 50  # Number of images (n) to average over.
	param_sigma_multiplier = 4
	smooth_grad = cnnvis.smooth_grad.generate_smooth_grad(GBP, img, target_class, param_n, param_sigma_multiplier)
	# Save colored gradients
	cnnvis_mf.save_gradient_images(smooth_grad, file_name_to_export + '_Guided_BP_SmoothGrad_color')

	# Convert to grayscale.
	grayscale_smooth_grad = cnnvis_mf.convert_to_grayscale(smooth_grad)
	# Save grayscale gradients.
	cnnvis_mf.save_gradient_images(grayscale_smooth_grad, file_name_to_export + '_Guided_BP_SmoothGrad_gray')

	print('Guided BackProp Smooth Grad completed: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Grad CAM.

	import pytorch_cnn_visualizations.gradcam

	start_time = time.time()
	grad_cam = cnnvis.gradcam.GradCam(pretrained_model, target_layer=11)

	# Generate CAM mask.
	cam = grad_cam.generate_cam(img, target_class)

	# Save mask.
	cnnvis_mf.save_class_activation_images(orig_img, cam, file_name_to_export)

	print('Grad CAM completed: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Guided Grad CAM.

	import pytorch_cnn_visualizations.guided_gradcam

	start_time = time.time()
	# Grad CAM.
	gcv2 = cnnvis.gradcam.GradCam(pretrained_model, target_layer=11)

	# Generate CAM mask.
	cam = gcv2.generate_cam(img, target_class)

	print('Grad CAM completed.')

	# Guided BackProp.
	GBP = cnnvis.guided_gradcam.GuidedBackprop(pretrained_model)
	# Get gradients.
	guided_grads = GBP.generate_gradients(img, target_class)

	print('Guided BackProp completed.')

	# Guided Grad CAM.
	cam_gb = cnnvis.guided_gradcam.guided_grad_cam(cam, guided_grads)
	cnnvis_mf.save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')

	grayscale_cam_gb = cnnvis_mf.convert_to_grayscale(cam_gb)
	cnnvis_mf.save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')

	print('Guided Grad CAM completed: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Integrated Gradients.

	import pytorch_cnn_visualizations.integrated_gradients

	start_time = time.time()
	IG = cnnvis.integrated_gradients.IntegratedGradients(pretrained_model)

	# Generate gradients.
	integrated_grads = IG.generate_integrated_gradients(img, target_class, 100)

	# Convert to grayscale.
	grayscale_integrated_grads = cnnvis_mf.convert_to_grayscale(integrated_grads)
	# Save grayscale gradients.
	cnnvis_mf.save_gradient_images(grayscale_integrated_grads, file_name_to_export + '_Integrated_G_gray')

	print('Integrated Gradients completed: {} secs.'.format(time.time() - start_time))

	# Gradient x Image.
	start_time = time.time()
	grad_times_image = integrated_grads[0] * img.detach().numpy()[0]
	# Convert to grayscale.
	grayscale_integrated_grads = cnnvis_mf.convert_to_grayscale(grad_times_image)
	# Save grayscale gradients.
	cnnvis_mf.save_gradient_images(grayscale_integrated_grads, file_name_to_export + '_Integrated_G_times_image_gray')

	print('Integrated Gradients times image completed: {} secs.'.format(time.time() - start_time))

# REF [file] >> ${pytorch-cnn-visualizations_HOME}/cnn_layer_visualization.py
def cnn_filter_visualization_example():
	import pytorch_cnn_visualizations.cnn_layer_visualization

	start_time = time.time()
	#cnn_layer = 2  # Conv 1-2.
	#cnn_layer = 10  # Conv 2-1.
	cnn_layer = 17  # Conv 3-1.
	#cnn_layer = 24  # Conv 4-1.
	filter_pos = 5

	# Fully connected layer is not needed.
	pretrained_model = torchvision.models.vgg16(pretrained=True).features

	layer_vis = cnnvis.cnn_layer_visualization.CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

	# Layer visualization with pytorch hooks.
	layer_vis.visualise_layer_with_hooks()

	# Layer visualization without pytorch hooks.
	#layer_vis.visualise_layer_without_hooks()

	print('CNN filter visualization completed: {} secs.'.format(time.time() - start_time))

# REF [file] >> ${pytorch-cnn-visualizations_HOME}/layer_activation_with_guided_backprop.py
def layer_activation_visualization_example():
	import pytorch_cnn_visualizations.layer_activation_with_guided_backprop

	# Pick one of the examples.
	img_path, target_class = './snake.jpg', 56
	#img_path, target_class = './cat_dog.png', 243
	#img_path, target_class = './spider.png', 72

	file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]

	# Read image.
	orig_img = PIL.Image.open(img_path).convert('RGB')
	# Process image.
	img = cnnvis_mf.preprocess_image(orig_img)

	# Define model.
	pretrained_model = torchvision.models.alexnet(pretrained=True)

	#--------------------
	# Layer activation with Guided BackProp.
	start_time = time.time()

	# Guided backprop.
	GBP = cnnvis.layer_activation_with_guided_backprop.GuidedBackprop(pretrained_model)

	filter_pos = 0
	for cnn_layer in range(30):
		fname = file_name_to_export + '_layer' + str(cnn_layer) + '_filter' + str(filter_pos)

		# Get gradients.
		guided_grads = GBP.generate_gradients(img, target_class, cnn_layer, filter_pos)
		# Save colored gradients.
		cnnvis_mf.save_gradient_images(guided_grads, fname + '_Guided_BP_color_layer_vis')

		# Convert to grayscale.
		grayscale_guided_grads = cnnvis_mf.convert_to_grayscale(guided_grads)
		# Save grayscale gradients.
		cnnvis_mf.save_gradient_images(grayscale_guided_grads, fname + '_Guided_BP_gray_layer_vis')

		# Positive and negative saliency maps.
		pos_sal, neg_sal = cnnvis_mf.get_positive_negative_saliency(guided_grads)
		cnnvis_mf.save_gradient_images(pos_sal, fname + '_pos_sal_layer_vis')
		cnnvis_mf.save_gradient_images(neg_sal, fname + '_neg_sal_layer_vis')

	#cnn_layer = 2  # Conv 1-2.
	#cnn_layer = 10  # Conv 2-1.
	#cnn_layer = 17  # Conv 3-1.
	#cnn_layer = 24  # Conv 4-1.
	cnn_layer = 29
	for filter_pos in range(30):
		fname = file_name_to_export + '_layer' + str(cnn_layer) + '_filter' + str(filter_pos)

		# Get gradients.
		guided_grads = GBP.generate_gradients(img, target_class, cnn_layer, filter_pos)
		# Save colored gradients.
		cnnvis_mf.save_gradient_images(guided_grads, fname + '_Guided_BP_color_layer_vis')

		# Convert to grayscale.
		grayscale_guided_grads = cnnvis_mf.convert_to_grayscale(guided_grads)
		# Save grayscale gradients.
		cnnvis_mf.save_gradient_images(grayscale_guided_grads, fname + '_Guided_BP_gray_layer_vis')

		# Positive and negative saliency maps.
		pos_sal, neg_sal = cnnvis_mf.get_positive_negative_saliency(guided_grads)
		cnnvis_mf.save_gradient_images(pos_sal, fname + '_pos_sal_layer_vis')
		cnnvis_mf.save_gradient_images(neg_sal, fname + '_neg_sal_layer_vis')

	print('Layer Guided BackProp completed: {} secs.'.format(time.time() - start_time))

# REF [file] >> ${pytorch-cnn-visualizations_HOME}/inverted_representation.py
def inverted_image_representation_example():
	import pytorch_cnn_visualizations.inverted_representation

	# Pick one of the examples.
	img_path, target_class = './snake.jpg', 56
	#img_path, target_class = './cat_dog.png', 243
	#img_path, target_class = './spider.png', 72

	file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]

	# Read image.
	orig_img = PIL.Image.open(img_path).convert('RGB')
	# Process image.
	img = cnnvis_mf.preprocess_image(orig_img)

	# Define model.
	pretrained_model = torchvision.models.alexnet(pretrained=True)

	#--------------------
	inverted_representation = cnnvis.inverted_representation.InvertedRepresentation(pretrained_model)

	image_size = 224  # Width & height.
	target_layer = 4
	inverted_representation.generate_inverted_image_specific_layer(img, image_size, target_layer)

# REF [file] >> ${pytorch-cnn-visualizations_HOME}/deep_dream.py
def deep_dream_example():
	import pytorch_cnn_visualizations.deep_dream

	# Fully connected layer is not needed.
	pretrained_model = torchvision.models.vgg19(pretrained=True).features

	cnn_layer = 34
	filter_pos = 94
	img_path = './dd_tree.jpg'
	dd = cnnvis.deep_dream.DeepDream(pretrained_model, cnn_layer, filter_pos, img_path)

	# This operation can also be done without Pytorch hooks.
	# See layer visualisation for the implementation without hooks.
	dd.dream()

# REF [file] >> ${pytorch-cnn-visualizations_HOME}/generate_class_specific_samples.py
def generate_class_specific_samples_example():
	import pytorch_cnn_visualizations.generate_class_specific_samples

	pretrained_model = torchvision.models.alexnet(pretrained=True)

	target_class = 130  # Flamingo.
	csig = cnnvis.generate_class_specific_samples.ClassSpecificImageGeneration(pretrained_model, target_class)
	csig.generate()

def main():
	gradient_visualization_example()

	#cnn_filter_visualization_example()
	#layer_activation_visualization_example()
	#inverted_image_representation_example()

	#deep_dream_example()

	#generate_class_specific_samples_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
