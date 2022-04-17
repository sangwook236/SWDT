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
	raise NotImplementedError

# REF [site] >> https://voxel51.com/docs/fiftyone/user_guide/app.html
def app_test():
	dataset = foz.load_zoo_dataset("quickstart")
	#dataset = foz.load_zoo_dataset("cifar10")

	session = fo.launch_app(dataset, port=5151)
	#session.show()

	# Blocks execution until the App is closed.
	session.wait()

	# View the dataset in the App.
	session.dataset = dataset

def main():
	#dataset_test()
	#dataset_view_test()  # Not yet implemented.

	app_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
