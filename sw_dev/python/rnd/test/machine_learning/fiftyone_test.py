#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import itertools
from datetime import datetime, timedelta
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
#import fiftyone.brain as fob  # FIXME [error] >> Segmentation fault.

# REF [site] >>
#	https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/index.html
#	https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html
def dataset_test():
	if True:
		my_dataset_name = "quickstart"
		if my_dataset_name in fo.list_datasets():
			dataset = fo.load_dataset(my_dataset_name)
		else:
			dataset = foz.load_zoo_dataset(my_dataset_name)
	elif False:
		dataset = foz.load_zoo_dataset(
			"coco-2017",
			split="validation",
			dataset_name="evaluate-detections-tutorial",
		)
	elif False:
		dataset = foz.load_zoo_dataset(
			"open-images-v6",
			split="validation",
			max_samples=100,
			seed=51,
			shuffle=True,
		)
	elif True:
		print("Datasets = {}.".format(fo.list_datasets()))

		my_dataset_name = "my_dataset"
		try:
			# REF [site] >> https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html
			if True:
				dataset_dir_path = "/path/to/data"
				dataset = fo.Dataset.from_dir(
					dataset_dir=dataset_dir_path,
					dataset_type=fo.types.ImageClassificationDirectoryTree,
					name=my_dataset_name,
				)
			elif False:
				# The directory containing the source images.
				data_dir_path = "/path/to/images"
				# The path to the COCO labels JSON file.
				label_filepath = "/path/to/coco-labels.json"
				dataset = fo.Dataset.from_dir(
					dataset_type=fo.types.COCODetectionDataset,
					data_path=data_dir_path,
					labels_path=label_filepath,
				)
		except ValueError:
			dataset = fo.load_dataset(my_dataset_name)

		#fo.delete_dataset(my_dataset_name)
	elif False:
		if True:
			# Create a dataset from a list of images.
			dataset = fo.Dataset.from_images(
				["/path/to/image1.jpg", "/path/to/image2.jpg",]
			)
		elif False:
			# Create a dataset from a directory of images.
			dataset = fo.Dataset.from_images_dir("/path/to/images")
		elif False:
			# Create a dataset from a glob pattern of images.
			dataset = fo.Dataset.from_images_patt("/path/to/images/*.jpg")
	elif False:
		if True:
			# Create a dataset from a list of videos
			dataset = fo.Dataset.from_videos(
				["/path/to/video1.mp4", "/path/to/video2.mp4",]
			)
		elif False:
			# Create a dataset from a directory of videos.
			dataset = fo.Dataset.from_videos_dir("/path/to/videos")
		elif False:
			# Create a dataset from a glob pattern of videos.
			dataset = fo.Dataset.from_videos_patt("/path/to/videos/*.mp4")
	dataset.persistent = True

	print("Media type = {}.".format(dataset.media_type))
	print("Persistence = {}.".format(dataset.persistence))
	print("#examples = {}.".format(len(dataset)))
	#print("#examples = {}.".format(dataset.count()))

	# Print some information about the dataset.
	print(dataset)

	# Print a ground truth detection.
	sample = dataset.first()
	if sample.ground_truth and hasattr(sample.ground_truth, "detections"):
		print(sample.ground_truth.detections[0])

	#--------------------
	#results = fob.compute_similarity(dataset, brain_key="img_sim")
	#results.find_unique(100)

	#vis_results = fob.compute_visualization(dataset, brain_key="img_vis")
	#plot = results.visualize_unique(visualization=vis_results)
	#plot.show()

# REF [site] >> https://voxel51.com/docs/fiftyone/user_guide/using_views.html
def dataset_view_test():
	dataset = foz.load_zoo_dataset("quickstart")

	view = dataset.view()

	print(view)

	print("Media type = {}.".format(view.media_type))
	print("#examples = {}.".format(len(view)))
	#print("#examples = {}.".format(view.count()))

	#--------------------
	#for sample in view:
	#	print(sample)

	#--------------------
	sample = view.take(1).first()

	print(type(sample))  # fiftyone.core.sample.SampleView.

	same_sample = view[sample.id]
	also_same_sample = view[sample.filepath]

	#view[other_sample_id]  # KeyError: sample non-existent or not in view.

	# List available view operations on a dataset.
	print(dataset.list_view_stages())

	#--------------------
	# View stages.

	# Random set of 100 samples from the dataset
	random_view = dataset.take(100)

	print("#examples = {}.".format(len(random_view)))

	# Sort 'random_view' by filepath.
	sorted_random_view = random_view.sort_by("filepath")

	#--------------------
	# Slicing.

	# Skip the first 2 samples and take the next 3.
	range_view1 = dataset.skip(2).limit(3)

	# Equivalently, using array slicing.
	range_view2 = dataset[2:5]

	view = dataset[10:100]

	sample10 = view.first()
	sample100 = view.last()

	also_sample10 = view[sample10.id]
	assert also_sample10.filepath == sample10.filepath

	also_sample100 = view[sample100.filepath]
	assert sample100.id == also_sample100.id

	assert sample10 is not also_sample10

	# A boolean array encoding the samples to extract.
	bool_array = np.array(dataset.values("uniqueness")) > 0.7
	view = dataset[bool_array]
	print("#examples = {}.".format(len(view)))

	ids = itertools.compress(dataset.values("id"), bool_array)
	view = dataset.select(ids)
	print("#examples = {}.".format(len(view)))

	# ViewExpression defining the samples to match.
	expr = fo.ViewField("uniqueness") > 0.7

	# Use a match() expression to define the view.
	view = dataset.match(expr)
	print("#examples = {}.".format(len(view)))

	# Equivalent: using boolean expression indexing is allowed too.
	view = dataset[expr]
	print("#examples = {}.".format(len(view)))

	#--------------------
	# Sorting.

	view = dataset.sort_by("filepath")
	view = dataset.sort_by("filepath", reverse=True)

	# Sort by number of detections in 'Detections' field 'ground_truth'.
	view = dataset.sort_by(fo.ViewField("ground_truth.detections").length(), reverse=True)

	print(len(view.first().ground_truth.detections))
	print(len(view.last().ground_truth.detections))

	#--------------------
	# Shuffling.

	# Randomly shuffle the order of the samples in the dataset.
	view1 = dataset.shuffle()

	# Randomly shuffle the samples in the dataset with a fixed seed.
	view2 = dataset.shuffle(seed=51)
	print(view2.first().id)

	also_view2 = dataset.shuffle(seed=51)
	print(also_view2.first().id)

	#--------------------
	# Random sampling.

	# Take 5 random samples from the dataset.
	view1 = dataset.take(5)

	# Take 5 random samples from the dataset with a fixed seed.
	view2 = dataset.take(5, seed=51)
	print(view2.first().id)

	also_view2 = dataset.take(5, seed=51)
	print(also_view2.first().id)

	#--------------------
	# Filtering.

	# Populate metadata on all samples.
	dataset.compute_metadata()

	# Samples whose image is less than 48 KB.
	small_images_view = dataset.match(fo.ViewField("metadata.size_bytes") < 48 * 1024)

	# Samples that contain at least one prediction with confidence above 0.99 or whose label ifs "cat" or "dog".
	match = (fo.ViewField("confidence") > 0.99) | (fo.ViewField("label").is_in(("cat", "dog")))
	matching_view = dataset.match(fo.ViewField("predictions.detections").filter(match).length() > 0)

	# The validation split of the dataset.
	val_view = dataset.match_tags("validation")
	# Union of the validation and test splits.
	val_test_view = dataset.match_tags(("validation", "test"))
	# The subset of samples where predictions have been computed.
	predictions_view = dataset.exists("predictions")

	# Get the IDs of two random samples.
	sample_ids = [
		dataset.take(1).first().id,
		dataset.take(1).first().id,
	]
	# Include only samples with the given IDs in the view.
	selected_view = dataset.select(sample_ids)
	# Exclude samples with the given IDs from the view.
	excluded_view = dataset.exclude(sample_ids)

	for sample in dataset.select_fields("ground_truth"):
		print(sample.id)            # OKAY: 'id' is always available
		print(sample.ground_truth)  # OKAY: 'ground_truth' was selected
		#print(sample.predictions)   # AttributeError: 'predictions' was not selected

	for sample in dataset.exclude_fields("predictions"):
		print(sample.id)            # OKAY: 'id' is always available
		print(sample.ground_truth)  # OKAY: 'ground_truth' was not excluded
		#print(sample.predictions)   # AttributeError: 'predictions' was excluded

	#--------------------
	# Date-based views.

	dataset = fo.Dataset()
	dataset.add_samples(
		[
			fo.Sample(
				filepath="image1.png",
				capture_date=datetime(2021, 8, 24, 1, 0, 0),
			),
			fo.Sample(
				filepath="image2.png",
				capture_date=datetime(2021, 8, 24, 2, 0, 0),
			),
			fo.Sample(
				filepath="image3.png",
				capture_date=datetime(2021, 8, 24, 3, 0, 0),
			),
		]
	)

	query_date = datetime(2021, 8, 24, 2, 1, 0)
	query_delta = timedelta(minutes=30)

	# Samples with capture date after 2021-08-24 02:01:00.
	view = dataset.match(fo.ViewField("capture_date") > query_date)
	print(view)

	# Samples with capture date within 30 minutes of 2021-08-24 02:01:00.
	view = dataset.match(abs(fo.ViewField("capture_date") - query_date) < query_delta)
	print(view)

# REF [site] >> https://voxel51.com/docs/fiftyone/user_guide/app.html
def app_test():
	dataset = foz.load_zoo_dataset("quickstart")
	#dataset = foz.load_zoo_dataset("quickstart-video")
	#dataset = foz.load_zoo_dataset("cifar10")

	session = fo.launch_app(dataset, port=5151)
	#session.show()

	if False:
		# View the dataset in the App.
		session.dataset = dataset
	elif False:
		# Object patches.

		# Convert to ground truth patches.
		gt_patches = dataset.to_patches("ground_truth")
		print(gt_patches)

		# View patches in the App.
		session.view = gt_patches
	elif False:
		# Evaluation patches.

		# Evaluate 'predictions' w.r.t. labels in 'ground_truth' field.
		dataset.evaluate_detections("predictions", gt_field="ground_truth", eval_key="eval")

		# Convert to evaluation patches.
		eval_patches = dataset.to_evaluation_patches("eval")
		print(eval_patches)

		print(eval_patches.count_values("type"))

		# View patches in the App.
		session.view = eval_patches

	# Blocks execution until the App is closed.
	session.wait()

def main():
	#dataset_test()
	#dataset_view_test()

	app_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
