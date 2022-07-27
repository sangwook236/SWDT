#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')

import random, time
import numpy as np
import augraphy
import cv2
import matplotlib.pyplot as plt

# REF [site] >> https://github.com/sparkfish/augraphy
def simple_example():
	image_filepath = './AugraphyExampleInput.png'
	#image_filepath = './Text.png'

	img = cv2.imread(image_filepath)
	if img is None:
		print('Image file not found, {}.'.format(image_filepath))
		return

	print('Augmenting...')
	start_time = time.time()
	augmented = augraphy.default_augraphy_pipeline(img)
	#pipeline = augraphy.default_augraphy_pipeline()
	#augmented = pipeline.augment(img)
	#augmented = augraphy.default_augment(img)
	print('Augmented: {} secs.'.format(time.time() - start_time))
	augmented = augmented['output']

	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
	ax[0].imshow(img)
	ax[0].set_title('Input')
	ax[0].axis('off')
	ax[1].imshow(augmented)
	ax[1].set_title('Augment')
	ax[1].axis('off')
	plt.tight_layout()
	plt.show()

def get_custom_pipeline():
	dithering_dither_type = random.choice(['ordered', 'floyd-steinberg'])
	dithering_order = random.choice(range(3, 10))
	dithering_p = (random.random() > 0.7) * 1
	inkbleed_intensity_range = (0.1, 0.4)
	inkbleed_color_range = (0, 32)
	inkbleed_kernel_size = random.choice([(7, 7), (5, 5), (3, 3)])
	inkbleed_severity = (0.4, 0.6)
	bleedthrough_intensity_range = (0.1, 0.2)
	bleedthrough_color_range = (32, 224)
	bleedthrough_ksize = (17, 17)
	bleedthrough_sigmaX = 0
	bleedthrough_alpha = random.uniform(0.05, 0.1)
	bleedthrough_offsets = (10, 20)
	letterpress_n_samples = (100, 300)
	letterpress_n_clusters = (300, 400)
	letterpress_std_range = (500, 3000)
	letterpress_value_range = (150, 224)
	letterpress_value_threshold_range = (96, 128)
	letterpress_blur = 1
	letterpress_p = (random.random() > 0.5) * 1

	lowinkrandomlines_count_range = (3, 12)
	lowinkperiodiclines_count_range = (1, 2)
	lowinkrandomlines_use_consistent_lines = random.choice([True, False])
	lowinkperiodiclines_use_consistent_lines = random.choice([True, False])
	lowinkperiodiclines_period_range = (16, 32)

	paperfactory_texture_path = './paper_textures'

	noisetexturize_sigma_range = (3, 10)
	noisetexturize_turbulence_range = (2, 5)
	
	brightnesstexturize_range = (0.9, 0.99)
	brightnesstexturize_deviation = 0.03

	brightness_range = (0.9, 1.1)

	pageborder_side = random.choice(['left', 'top', 'bottom', 'right'])
	pageborder_width_range = (5, 30)

	pageborder_pages = None
	pageborder_intensity_range = (0.4, 0.8)
	pageborder_curve_frequency = (2, 8)
	pageborder_curve_height = (2, 4)
	pageborder_curve_length_one_side = (50, 100)
	pageborder_value = (32, 150)
	dirtyrollers_line_width_range = (2, 32)
	dirtyrollers_scanline_type = 0
	dirtyrollers_p = (random.random() > 0.5) * 1
	lightinggradient_light_position = None
	lightinggradient_direction = None
	lightinggradient_max_brightness = 196
	lightinggradient_min_brightness = 0
	lightinggradient_mode = random.choice(['linear_dynamic', 'linear_static', 'gaussian'])
	lightinggradient_linear_decay_rate = None
	lightinggradient_transparency = None
	dirtydrum_line_width_range = (1, 6)
	dirtydrum_line_concentration = np.random.uniform(0.05, 0.15)
	dirtydrum_direction = random.randint(0, 2)
	dirtydrum_noise_intensity = np.random.uniform(0.6, 0.95)
	dirtydrum_noise_value = (128, 224)
	dirtydrum_ksize = random.choice([(3, 3), (5, 5), (7, 7)])
	dirtydrum_sigmaX = 0
	dirtydrum_p = (random.random() > 0.5) * 1
	subtlenoise_range = 10
	jpeg_quality_range = (25, 95)
	markup_num_lines_range = (2, 7)
	markup_length_range = (0.5, 1)
	markup_thickness_range = (1, 2)
	markup_type = random.choice(['strikethrough', 'crossed', 'highlight', 'underline'])
	markup_color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
	markup_single_word_mode = random.choice([True, False])
	markup_repetitions = random.randint(1, 2) if markup_type == 'highlight' else 1
	markup_p = (random.random() > 0.5) * 1
	pencilscribbles_size_range = (100, 700)
	pencilscribbles_count_range = (1, 6)
	pencilscribbles_stroke_count_range = (1, 1)
	pencilscribbles_thickness_range = (2, 6)
	pencilscribbles_brightness_change = random.randint(64, 224)
	pencilscribbles_p = (random.random() > 0.5) * 1
	bindingsandfasteners_overlay_types = 'darken'
	bindingsandfasteners_foreground = None
	bindingsandfasteners_effect_type = random.choice(['punch_holes', 'binding_holes', 'clips'])
	bindingsandfasteners_ntimes = (10, 20) if bindingsandfasteners_effect_type == 'binding_holes' else (2, 3)
	bindingsandfasteners_nscales = (0.9, 1)
	bindingsandfasteners_edge = 'random'
	bindingsandfasteners_edge_offset = (10, 50)
	badphotocopy_mask = None
	badphotocopy_noise_type = -1
	badphotocopy_noise_side = 'random'
	badphotocopy_noise_iteration = (1, 2)
	badphotocopy_noise_size = (1, 2)
	badphotocopy_noise_value = (128, 196)
	badphotocopy_noise_sparsity = (0.3, 0.6)
	badphotocopy_noise_concentration = (0.1, 0.5)
	badphotocopy_blur_noise = random.choice([True, False])
	badphotocopy_blur_noise_kernel = random.choice([(3, 3), (5, 5), (7, 7)])
	badphotocopy_wave_pattern = random.choice([True, False])
	badphotocopy_edge_effect = random.choice([True, False])
	badphotocopy_p = (random.random() > 0.5) * 1

	gamma_range = (0.8, 1.2)
	geometric_scale = (1, 1)
	geometric_translation = (0, 0)
	geometric_fliplr = 0
	geometric_flipud = 0
	geometric_crop = ()
	geometric_rotate_range = (0, 0)

	faxify_scale_range = (0.3, 0.6)
	faxify_monochrome = random.choice([0, 1])
	faxify_monochrome_method = 'random'
	faxify_monochrome_arguments = {}
	faxify_halftone = random.choice((0, 1))
	faxify_invert = 1
	faxify_half_kernel_size = (1, 1)
	faxify_angle = (0, 360)
	faxify_sigma = (1, 3)

	if badphotocopy_p > 0 or dirtyrollers_p > 0 or dirtydrum_p > 0 or markup_p > 0 or pencilscribbles_p > 0:
		faxify_monochrome_method = 'grayscale'
	if dithering_p or faxify_halftone:
		letterpress_p = 0
		if dithering_p:
			faxify_halftone = 0

	ink_phase = [
		augraphy.Dithering(dithering_dither_type,
			dithering_order,
			p=dithering_p
		),

		augraphy.InkBleed(inkbleed_intensity_range,
			inkbleed_color_range,
			inkbleed_kernel_size,
			inkbleed_severity,
			p=0.5
		),

		augraphy.BleedThrough(bleedthrough_intensity_range,
			bleedthrough_color_range,
			bleedthrough_ksize,
			bleedthrough_sigmaX,
			bleedthrough_alpha,
			bleedthrough_offsets,
			p=0.5
		),

		augraphy.Letterpress(letterpress_n_samples,
			letterpress_n_clusters,
			letterpress_std_range,
			letterpress_value_range,
			letterpress_value_threshold_range,
			letterpress_blur,
			p=letterpress_p
		),

		augraphy.OneOf([
			augraphy.LowInkRandomLines(lowinkrandomlines_count_range,
				lowinkrandomlines_use_consistent_lines
			),

			augraphy.LowInkPeriodicLines(lowinkperiodiclines_count_range,
				lowinkperiodiclines_period_range,
				lowinkperiodiclines_use_consistent_lines
			),
		]),
	]

	paper_phase = [
		augraphy.PaperFactory(paperfactory_texture_path, p=0.5),
		augraphy.OneOf([
			augraphy.AugmentationSequence([
				augraphy.NoiseTexturize(noisetexturize_sigma_range, noisetexturize_turbulence_range),
				augraphy.BrightnessTexturize(brightnesstexturize_range, brightnesstexturize_deviation),
			]),
			augraphy.AugmentationSequence([
				augraphy.BrightnessTexturize(brightnesstexturize_range, brightnesstexturize_deviation),
				augraphy.NoiseTexturize(noisetexturize_sigma_range, noisetexturize_turbulence_range),
			]),
		], p=0.5),

		augraphy.Brightness(brightness_range, p=0.5),
	]
	
	post_phase = [
		augraphy.BrightnessTexturize(brightnesstexturize_range,
			brightnesstexturize_deviation,
			p=0.5
		),

		augraphy.OneOf([
			augraphy.PageBorder(pageborder_side,
				pageborder_width_range,
				pageborder_pages,
				pageborder_intensity_range,
				pageborder_curve_frequency,
				pageborder_curve_height,
				pageborder_curve_length_one_side,
				pageborder_value
			),

			augraphy.DirtyRollers(dirtyrollers_line_width_range,
				dirtyrollers_scanline_type,
				p=dirtyrollers_p
			)
		], p=0.5),

		augraphy.OneOf([
			augraphy.LightingGradient(lightinggradient_light_position,
				lightinggradient_direction,
				lightinggradient_max_brightness,
				lightinggradient_min_brightness,
				lightinggradient_mode,
				lightinggradient_linear_decay_rate,
				lightinggradient_transparency
			),

			augraphy.Brightness(brightness_range)
		], p=0.5),

		augraphy.DirtyDrum(dirtydrum_line_width_range,
			dirtydrum_line_concentration,
			dirtydrum_direction,
			dirtydrum_noise_intensity,
			dirtydrum_noise_value,
			dirtydrum_ksize,
			dirtydrum_sigmaX,
			p=dirtydrum_p
		),

		augraphy.SubtleNoise(subtlenoise_range, p=0.5),

		augraphy.Jpeg(jpeg_quality_range, p=0.5),

		augraphy.Markup(markup_num_lines_range,
			markup_length_range,
			markup_thickness_range,
			markup_type,
			markup_color,
			markup_single_word_mode,
			markup_repetitions,
			p=markup_p
		),

		augraphy.PencilScribbles(pencilscribbles_size_range,
			pencilscribbles_count_range,
			pencilscribbles_stroke_count_range,
			pencilscribbles_thickness_range,
			pencilscribbles_brightness_change,
			p=pencilscribbles_p
		),

		augraphy.BindingsAndFasteners(bindingsandfasteners_overlay_types,
			bindingsandfasteners_foreground,
			bindingsandfasteners_effect_type,
			bindingsandfasteners_ntimes,
			bindingsandfasteners_nscales,
			bindingsandfasteners_edge,
			bindingsandfasteners_edge_offset,
			p=0.5
		),

		augraphy.BadPhotoCopy(badphotocopy_mask,
			badphotocopy_noise_type,
			badphotocopy_noise_side,
			badphotocopy_noise_iteration,
			badphotocopy_noise_size,
			badphotocopy_noise_value,
			badphotocopy_noise_sparsity,
			badphotocopy_noise_concentration,
			badphotocopy_blur_noise,
			badphotocopy_blur_noise_kernel,
			badphotocopy_wave_pattern,
			badphotocopy_edge_effect,
			p=badphotocopy_p
		),

		augraphy.Gamma(gamma_range, p=0.5),

		augraphy.Geometric(geometric_scale,
			geometric_translation,
			geometric_fliplr,
			geometric_flipud,
			geometric_crop,
			geometric_rotate_range,
			p=0.5
		),

		augraphy.Faxify(faxify_scale_range,
			faxify_monochrome,
			faxify_monochrome_method,
			faxify_monochrome_arguments,
			faxify_halftone,
			faxify_invert,
			faxify_half_kernel_size,
			faxify_angle,
			faxify_sigma,
			p=0.5
		),
	]

	return augraphy.AugraphyPipeline(ink_phase, paper_phase, post_phase, ink_color_range=(-1, -1), paper_color_range=(255, 255), log=False)

def custom_pipeline_test():
	image_filepaths = [
		'./AugraphyExampleInput.png',
		'./Text.png',
	]

	pipeline = get_custom_pipeline()

	# Visualize.
	fig, ax = plt.subplots(nrows=2, ncols=len(image_filepaths), figsize=(16, 16))
	for i, image_filepath in enumerate(image_filepaths):
		img = cv2.imread(image_filepath)
		if img is None:
			print('Image file not found, {}.'.format(image_filepath))
			continue

		print('Augmenting...')
		start_time = time.time()
		img_augmented = pipeline.augment(img)['output']
		print('Augmented: {} secs.'.format(time.time() - start_time))

		ax[0, i].imshow(img)
		ax[0, i].set_title('Input')
		ax[0, i].axis('off')
		ax[1, i].imshow(img_augmented)
		ax[1, i].set_title('Augment')
		ax[1, i].axis('off')

	plt.tight_layout()
	plt.show()

def main():
	#simple_example()
	custom_pipeline_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
