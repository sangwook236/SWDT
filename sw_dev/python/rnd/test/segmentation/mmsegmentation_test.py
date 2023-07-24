#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://mmsegmentation.readthedocs.io/en/latest/user_guides/3_inference.html
def inference_with_existing_models_tutorial():
	from mmseg.apis import MMSegInferencer

	# Load models into memory.
	inferencer = MMSegInferencer(model="deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024")

	# Inference.
	inferencer("./demo/demo.png", show=True)

	images = [image1, image2, image3]  # a np.ndarray.
	#images = ["/path/to/image1", "/path/to/image2", "/path/to/image3"]
	#images = "/path/to/image_dir"
	inferencer(images, show=True, wait_time=0.5)  # wait_time is delay time, and 0 means forever.

	# Save visualized rendering color maps and predicted results.
	# out_dir is the directory to save the output results, img_out_dir and pred_out_dir are subdirectories of out_dir to save visualized rendering color maps and predicted results.
	inferencer(images, out_dir="outputs", img_out_dir="vis", pred_out_dir="pred")

	#-----
	result = inferencer("./demo/demo.png")  # {'visualization', 'predictions'}.

	# 'visualization' includes color segmentation map.
	print(result["visualization"].shape)  # (512, 683, 3)
	# 'predictions' includes segmentation mask with label indice.
	print(result["predictions"].shape)  # (512, 683)

	result = inferencer("./demo/demo.png", return_datasamples=True)
	print(type(result))  # <class 'mmseg.structures.seg_data_sample.SegDataSample'>.

	# Input a list of images.
	results = inferencer(images)
	print(type(results["visualization"]), results["visualization"][0].shape)  # <class 'list'> (512, 683, 3).
	print(type(results["predictions"]), results["predictions"][0].shape)  # <class 'list'> (512, 683).

	results = inferencer(images, return_datasamples=True)
	print(type(results[0]))  # <class 'mmseg.structures.seg_data_sample.SegDataSample'>.

def main():
	inference_with_existing_models_tutorial()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
