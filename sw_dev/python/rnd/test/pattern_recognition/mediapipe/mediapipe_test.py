#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/object_detection/python/object_detector.ipynb
#	https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/object_detection/python/object_detector.ipynb
def object_detection_example():
	# Download:
	#	wget -O efficientdet.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
	#	wget https://storage.googleapis.com/mediapipe-tasks/object_detector/cat_and_dog.jpg

	import numpy as np
	import cv2
	import mediapipe as mp
	from mediapipe.tasks import python

	MARGIN = 10  # [pixels]
	ROW_SIZE = 10  # [pixels]
	FONT_SIZE = 1
	FONT_THICKNESS = 1
	TEXT_COLOR = (255, 0, 0)  # Red

	def visualize(
		image,
		detection_result
	) -> np.ndarray:
		"""Draws bounding boxes on the input image and return it.
		Args:
			image: The input RGB image.
			detection_result: The list of all "Detection" entities to be visualize.
		Returns:
			Image with bounding boxes.
		"""
		for detection in detection_result.detections:
			# Draw bounding_box
			bbox = detection.bounding_box
			start_point = bbox.origin_x, bbox.origin_y
			end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
			cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

			# Draw label and score
			category = detection.categories[0]
			category_name = category.category_name
			probability = round(category.score, 2)
			result_text = category_name + " (" + str(probability) + ")"
			text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
			cv2.putText(
				image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
				FONT_SIZE, TEXT_COLOR, FONT_THICKNESS,
			)

		return image

	IMAGE_FILE = "./cat_and_dog.jpg"

	img = cv2.imread(IMAGE_FILE)
	cv2.imshow("Image", img)

	# Create an ObjectDetector object
	base_options = python.BaseOptions(model_asset_path="./efficientdet.tflite")
	options = python.vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
	detector = python.vision.ObjectDetector.create_from_options(options)

	# Load the input image
	image = mp.Image.create_from_file(IMAGE_FILE)

	# Detect objects in the input image
	detection_result = detector.detect(image)

	# Process the detection result. In this case, visualize it
	image_copy = np.copy(image.numpy_view())
	annotated_image = visualize(image_copy, detection_result)
	rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
	cv2.imshow("Object Detection", rgb_annotated_image)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

# REF [site] >>
#	https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/python/image_classifier.ipynb
#	https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/image_classification/python/image_classifier.ipynb
def image_classification_example():
	# Download:
	#	wget wget -O classifier.tflite -q https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite
	#	wget https://storage.googleapis.com/mediapipe-tasks/image_classifier/burger.jpg
	#	wget https://storage.googleapis.com/mediapipe-tasks/image_classifier/cat.jpg

	if False:
		import urllib

		IMAGE_FILENAMES = ["burger.jpg", "cat.jpg"]
		for name in IMAGE_FILENAMES:
			url = f"https://storage.googleapis.com/mediapipe-tasks/image_classifier/{name}"
			urllib.request.urlretrieve(url, name)

	import math
	import cv2
	import mediapipe as mp
	from mediapipe.tasks import python
	import matplotlib.pyplot as plt

	plt.rcParams.update({
		"axes.spines.top": False,
		"axes.spines.right": False,
		"axes.spines.left": False,
		"axes.spines.bottom": False,
		"xtick.labelbottom": False,
		"xtick.bottom": False,
		"ytick.labelleft": False,
		"ytick.left": False,
		"xtick.labeltop": False,
		"xtick.top": False,
		"ytick.labelright": False,
		"ytick.right": False
	})

	def display_one_image(image, title, subplot, titlesize=16):
		"""Displays one image along with the predicted category name and score."""
		plt.subplot(*subplot)
		plt.imshow(image)
		if len(title) > 0:
			plt.title(title, fontsize=int(titlesize), color="black", fontdict={"verticalalignment": "center"}, pad=int(titlesize / 1.5))
		return (subplot[0], subplot[1], subplot[2]+1)

	def display_batch_of_images(images, predictions):
		"""Displays a batch of images with the classifications."""
		# Images and predictions
		images = [image.numpy_view() for image in images]

		# Auto-squaring: this will drop data that does not fit into square or square-ish rectangle
		rows = int(math.sqrt(len(images)))
		cols = len(images) // rows

		# Size and spacing
		FIGSIZE = 13.0
		SPACING = 0.1
		subplot = (rows, cols, 1)
		if rows < cols:
			plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
		else:
			plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))

		# Display
		for i, (image, prediction) in enumerate(zip(images[:rows * cols], predictions[:rows * cols])):
			dynamic_titlesize = FIGSIZE * SPACING / max(rows, cols) * 40 + 3
			subplot = display_one_image(image, prediction, subplot, titlesize=dynamic_titlesize)

		# Layout
		plt.tight_layout()
		plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
		plt.show()

	IMAGE_FILENAMES = ["burger.jpg", "cat.jpg"]
	DESIRED_HEIGHT = 480
	DESIRED_WIDTH = 480

	def resize_and_show(image):
		h, w = image.shape[:2]
		if h < w:
			img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
		else:
			img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
		cv2.imshow("Image", img)

	# Preview the images
	images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
	for name, image in images.items():
		print(name)
		resize_and_show(image)

	# Create an ImageClassifier object
	base_options = python.BaseOptions(model_asset_path="./classifier.tflite")
	options = python.vision.ImageClassifierOptions(base_options=base_options, max_results=4)
	classifier = python.vision.ImageClassifier.create_from_options(options)

	images = []
	predictions = []
	for image_name in IMAGE_FILENAMES:
		# Load the input image
		image = mp.Image.create_from_file(image_name)

		# Classify the input image
		classification_result = classifier.classify(image)

		# Process the classification result. In this case, visualize it
		images.append(image)
		top_category = classification_result.classifications[0].categories[0]
		predictions.append(f"{top_category.category_name} ({top_category.score:.2f})")

	display_batch_of_images(images, predictions)

# REF [site] >>
#	https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_segmentation/python/image_segmentation.ipynb
#	https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/image_segmentation/python/image_segmentation.ipynb
def image_segmentation_example():
	# Download:
	#	wget -O deeplabv3.tflite -q https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite
	#	wget https://storage.googleapis.com/mediapipe-assets/segmentation_input_rotation0.jpg

	if False:
		import urllib

		IMAGE_FILENAMES = ["segmentation_input_rotation0.jpg"]
		for name in IMAGE_FILENAMES:
			url = f"https://storage.googleapis.com/mediapipe-assets/{name}"
			urllib.request.urlretrieve(url, name)

	import math
	import numpy as np
	import cv2
	import mediapipe as mp
	from mediapipe.tasks import python

	IMAGE_FILENAMES = ["./segmentation_input_rotation0.jpg"]
	# Height and width that will be used by the model
	DESIRED_HEIGHT = 480
	DESIRED_WIDTH = 480

	# Performs resizing and showing the image
	def resize_and_show(image):
		h, w = image.shape[:2]
		if h < w:
			img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
		else:
			img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
		cv2.imshow("Image", img)

	# Preview the image(s)
	images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
	for name, image in images.items():
		print(name)
		resize_and_show(image)

	BG_COLOR = (192, 192, 192)  # Gray
	MASK_COLOR = (255, 255, 255)  # White

	# Create the options that will be used for ImageSegmenter
	base_options = python.BaseOptions(model_asset_path="./deeplabv3.tflite")
	options = python.vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

	# Create the image segmenter
	with python.vision.ImageSegmenter.create_from_options(options) as segmenter:
		# Loop through demo image(s)
		for image_file_name in IMAGE_FILENAMES:
			# Create the MediaPipe image file that will be segmented
			image = mp.Image.create_from_file(image_file_name)

			# Retrieve the masks for the segmented image
			segmentation_result = segmenter.segment(image)
			category_mask = segmentation_result.category_mask

			# Generate solid color images for showing the output segmentation mask
			image_data = image.numpy_view()
			fg_image = np.zeros(image_data.shape, dtype=np.uint8)
			fg_image[:] = MASK_COLOR
			bg_image = np.zeros(image_data.shape, dtype=np.uint8)
			bg_image[:] = BG_COLOR

			condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
			output_image = np.where(condition, fg_image, bg_image)

			print(f"Segmentation mask of {name}:")
			resize_and_show(output_image)

	# Blur the image background based on the segmentation mask

	# Create the segmenter
	with python.vision.ImageSegmenter.create_from_options(options) as segmenter:
		# Loop through available image(s)
		for image_file_name in IMAGE_FILENAMES:

			# Create the MediaPipe Image
			image = mp.Image.create_from_file(image_file_name)

			# Retrieve the category masks for the image
			segmentation_result = segmenter.segment(image)
			category_mask = segmentation_result.category_mask

			# Convert the BGR image to RGB
			image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_BGR2RGB)

			# Apply effects
			blurred_image = cv2.GaussianBlur(image_data, (55,55), 0)
			condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
			output_image = np.where(condition, image_data, blurred_image)

			print(f"Blurred background of {image_file_name}:")
			resize_and_show(output_image)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

# REF [site] >>
#	https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
#	https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
def hand_landmarks_detection_example():
	# Download:
	#	wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
	#	wget https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg

	import numpy as np
	import cv2
	import mediapipe as mp
	from mediapipe.tasks import python
	from mediapipe.framework.formats import landmark_pb2

	MARGIN = 10  # [pixels]
	FONT_SIZE = 1
	FONT_THICKNESS = 1
	HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # Vibrant green

	def draw_landmarks_on_image(rgb_image, detection_result):
		hand_landmarks_list = detection_result.hand_landmarks
		handedness_list = detection_result.handedness
		annotated_image = np.copy(rgb_image)

		# Loop through the detected hands to visualize
		for idx in range(len(hand_landmarks_list)):
			hand_landmarks = hand_landmarks_list[idx]
			handedness = handedness_list[idx]

			# Draw the hand landmarks
			hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
			hand_landmarks_proto.landmark.extend([
				landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
			])
			mp.solutions.drawing_utils.draw_landmarks(
				annotated_image,
				hand_landmarks_proto,
				mp.solutions.hands.HAND_CONNECTIONS,
				mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
				mp.solutions.drawing_styles.get_default_hand_connections_style(),
			)

			# Get the top left corner of the detected hand's bounding box
			height, width, _ = annotated_image.shape
			x_coordinates = [landmark.x for landmark in hand_landmarks]
			y_coordinates = [landmark.y for landmark in hand_landmarks]
			text_x = int(min(x_coordinates) * width)
			text_y = int(min(y_coordinates) * height) - MARGIN

			# Draw handedness (left or right hand) on the image
			cv2.putText(
				annotated_image, f"{handedness[0].category_name}",
				(text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
				FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA,
			)

		return annotated_image

	img = cv2.imread("./woman_hands.jpg")
	cv2.imshow("Hand Landmarks", img)

	# Create an HandLandmarker object
	base_options = python.BaseOptions(model_asset_path="./hand_landmarker.task")
	options = python.vision.HandLandmarkerOptions(
		base_options=base_options,
		num_hands=2,
	)
	detector = python.vision.HandLandmarker.create_from_options(options)

	# Load the input image
	image = mp.Image.create_from_file("./woman_hands.jpg")

	# Detect hand landmarks from the input image
	detection_result = detector.detect(image)

	# Process the classification result. In this case, visualize it
	annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
	cv2.imshow("Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

	cv2.waitKey(0)
	cv2.destroyAllWindows()

# REF [site] >> https://velog.io/@thithi250696/Deep-Learning-mediapipe-hand-사용하기-Hand-Landmark-가위바위보
def hand_landmark_detection_test():
	import cv2
	import mediapipe as mp

	detector = mp.solutions.hands.Hands(
		max_num_hands=2,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5,
	)

	video = cv2.VideoCapture(0)
	while video.isOpened():
		ret, img = video.read()
		if not ret:
			continue

		img = cv2.flip(img, 1)
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		detection_result = detector.process(img_rgb)  # Detect hands
		if detection_result.multi_hand_landmarks is not None:
			#print(detection_result.multi_hand_landmarks)
			for landmark in detection_result.multi_hand_landmarks:
				mp.solutions.drawing_utils.draw_landmarks(
					img,
					landmark_list=landmark,
					connections=mp.solutions.hands.HAND_CONNECTIONS,
					#landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
					landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=mp.solutions.drawing_utils.RED_COLOR),
					#connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style(),
    				connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=mp.solutions.drawing_utils.BLUE_COLOR),
					is_drawing_landmarks=True,
				)

		key = cv2.waitKey(10)
		if key == 27:
			break

		cv2.imshow("Hand Landmarks", img)

	video.release()
	cv2.destroyAllWindows()

# REF [site] >>
#	https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb
#	https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb
def gesture_recognition_example():
	# Download:
	#	wget https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
	#	wget https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_down.jpg
	#	wget https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/victory.jpg
	#	wget https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_up.jpg
	#	wget https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/pointing_up.jpg

	if False:
		import urllib

		IMAGE_FILENAMES = ["thumbs_down.jpg", "victory.jpg", "thumbs_up.jpg", "pointing_up.jpg"]
		for name in IMAGE_FILENAMES:
			url = f"https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/{name}"
			urllib.request.urlretrieve(url, name)

	import math
	import cv2
	import mediapipe as mp
	from mediapipe.tasks import python
	from mediapipe.framework.formats import landmark_pb2
	import matplotlib.pyplot as plt

	plt.rcParams.update({
		"axes.spines.top": False,
		"axes.spines.right": False,
		"axes.spines.left": False,
		"axes.spines.bottom": False,
		"xtick.labelbottom": False,
		"xtick.bottom": False,
		"ytick.labelleft": False,
		"ytick.left": False,
		"xtick.labeltop": False,
		"xtick.top": False,
		"ytick.labelright": False,
		"ytick.right": False
	})

	mp_hands = mp.solutions.hands
	mp_drawing = mp.solutions.drawing_utils
	mp_drawing_styles = mp.solutions.drawing_styles

	def display_one_image(image, title, subplot, titlesize=16):
		"""Displays one image along with the predicted category name and score."""
		plt.subplot(*subplot)
		plt.imshow(image)
		if len(title) > 0:
			plt.title(title, fontsize=int(titlesize), color="black", fontdict={"verticalalignment": "center"}, pad=int(titlesize / 1.5))
		return (subplot[0], subplot[1], subplot[2]+1)

	def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
		"""Displays a batch of images with the gesture category and its score along with the hand landmarks."""
		# Images and labels
		images = [image.numpy_view() for image in images]
		gestures = [top_gesture for (top_gesture, _) in results]
		multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

		# Auto-squaring: this will drop data that does not fit into square or square-ish rectangle
		rows = int(math.sqrt(len(images)))
		cols = len(images) // rows

		# Size and spacing
		FIGSIZE = 13.0
		SPACING = 0.1
		subplot=(rows, cols, 1)
		if rows < cols:
			plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
		else:
			plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))

		# Display gestures and hand landmarks
		for i, (image, gestures) in enumerate(zip(images[:rows * cols], gestures[:rows * cols])):
			title = f"{gestures.category_name} ({gestures.score:.2f})"
			dynamic_titlesize = FIGSIZE * SPACING / max(rows, cols) * 40 + 3
			annotated_image = image.copy()

			for hand_landmarks in multi_hand_landmarks_list[i]:
				hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
				hand_landmarks_proto.landmark.extend([
					landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
				])

				mp_drawing.draw_landmarks(
					annotated_image,
					hand_landmarks_proto,
					mp_hands.HAND_CONNECTIONS,
					mp_drawing_styles.get_default_hand_landmarks_style(),
					mp_drawing_styles.get_default_hand_connections_style(),
				)

			subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

		# Layout
		plt.tight_layout()
		plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
		plt.show()

	IMAGE_FILENAMES = ["thumbs_down.jpg", "victory.jpg", "thumbs_up.jpg", "pointing_up.jpg"]
	DESIRED_HEIGHT = 480
	DESIRED_WIDTH = 480

	def resize_and_show(image):
		h, w = image.shape[:2]
		if h < w:
			img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
		else:
			img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
		cv2.imshow("Image", img)

	# Preview the images
	images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
	for name, image in images.items():
		print(name)
		resize_and_show(image)

	# Create an GestureRecognizer object
	base_options = python.BaseOptions(model_asset_path="./gesture_recognizer.task")
	options = python.vision.GestureRecognizerOptions(base_options=base_options)
	recognizer = python.vision.GestureRecognizer.create_from_options(options)

	images = []
	results = []
	for image_file_name in IMAGE_FILENAMES:
		# Load the input image
		image = mp.Image.create_from_file(image_file_name)

		# Recognize gestures in the input image
		recognition_result = recognizer.recognize(image)

		# Process the result. In this case, visualize it
		images.append(image)
		top_gesture = recognition_result.gestures[0][0]
		hand_landmarks = recognition_result.hand_landmarks
		results.append((top_gesture, hand_landmarks))

	display_batch_of_images_with_gestures_and_hand_landmarks(images, results)

# REF [site] >>
#	https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_detector/python/face_detector.ipynb
#	https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_detector/python/face_detector.ipynb
def face_detection_example():
	# Download:
	#	wget -O detector.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
	#	wget https://i.imgur.com/Vu2Nqwb.jpeg

	import typing, math
	import numpy as np
	import cv2
	import mediapipe as mp
	from mediapipe.tasks import python

	MARGIN = 10  # [pixels]
	ROW_SIZE = 10  # [pixels]
	FONT_SIZE = 1
	FONT_THICKNESS = 1
	TEXT_COLOR = (255, 0, 0)  # Red

	def _normalized_to_pixel_coordinates(
		normalized_x: float, normalized_y: float, image_width: int,
		image_height: int) -> typing.Union[None, typing.Tuple[int, int]]:
		"""Converts normalized value pair to pixel coordinates."""

		# Checks if the float value is between 0 and 1
		def is_valid_normalized_value(value: float) -> bool:
			return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

		if not (is_valid_normalized_value(normalized_x) and
			is_valid_normalized_value(normalized_y)):
			# TODO: Draw coordinates even if it's outside of the image bounds
			return None
		x_px = min(math.floor(normalized_x * image_width), image_width - 1)
		y_px = min(math.floor(normalized_y * image_height), image_height - 1)
		return x_px, y_px

	def visualize(
		image,
		detection_result
	) -> np.ndarray:
		"""Draws bounding boxes and keypoints on the input image and return it.
		Args:
			image: The input RGB image.
			detection_result: The list of all "Detection" entities to be visualize.
		Returns:
			Image with bounding boxes.
		"""
		annotated_image = image.copy()
		height, width, _ = image.shape

		for detection in detection_result.detections:
			# Draw bounding_box
			bbox = detection.bounding_box
			start_point = bbox.origin_x, bbox.origin_y
			end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
			cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

			# Draw keypoints
			for keypoint in detection.keypoints:
				keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
				color, thickness, radius = (0, 255, 0), 2, 2
				cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

			# Draw label and score
			category = detection.categories[0]
			category_name = category.category_name
			category_name = "" if category_name is None else category_name
			probability = round(category.score, 2)
			result_text = category_name + " (" + str(probability) + ")"
			text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
			cv2.putText(
				annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
				FONT_SIZE, TEXT_COLOR, FONT_THICKNESS,
			)

		return annotated_image

	IMAGE_FILE = "./Vu2Nqwb.jpeg"

	img = cv2.imread(IMAGE_FILE)
	cv2.imshow("Image", img)

	# Create an FaceDetector object
	base_options = python.BaseOptions(model_asset_path="./detector.tflite")
	options = python.vision.FaceDetectorOptions(base_options=base_options)
	detector = python.vision.FaceDetector.create_from_options(options)

	# Load the input image
	image = mp.Image.create_from_file(IMAGE_FILE)

	# Detect faces in the input image
	detection_result = detector.detect(image)

	# Process the detection result. In this case, visualize it
	image_copy = np.copy(image.numpy_view())
	annotated_image = visualize(image_copy, detection_result)
	rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
	cv2.imshow("Face Detection", rgb_annotated_image)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

# REF [site]] >>
#	https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb
#	https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb
def face_landmarks_detection_example():
	# Download:
	#	wget -O face_landmarker_v2_with_blendshapes.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
	#	wget https://storage.googleapis.com/mediapipe-assets/business-person.png

	import numpy as np
	import cv2
	import mediapipe as mp
	from mediapipe.framework.formats import landmark_pb2
	from mediapipe.tasks import python
	import matplotlib.pyplot as plt

	def draw_landmarks_on_image(rgb_image, detection_result):
		face_landmarks_list = detection_result.face_landmarks
		annotated_image = np.copy(rgb_image)

		# Loop through the detected faces to visualize
		for idx in range(len(face_landmarks_list)):
			face_landmarks = face_landmarks_list[idx]

			# Draw the face landmarks
			face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
			face_landmarks_proto.landmark.extend([
				landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
			])

			mp.solutions.drawing_utils.draw_landmarks(
				image=annotated_image,
				landmark_list=face_landmarks_proto,
				connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
				landmark_drawing_spec=None,
				connection_drawing_spec=mp.solutions.drawing_styles
				.get_default_face_mesh_tesselation_style(),
			)
			mp.solutions.drawing_utils.draw_landmarks(
				image=annotated_image,
				landmark_list=face_landmarks_proto,
				connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
				landmark_drawing_spec=None,
				connection_drawing_spec=mp.solutions.drawing_styles
				.get_default_face_mesh_contours_style(),
			)
			mp.solutions.drawing_utils.draw_landmarks(
				image=annotated_image,
				landmark_list=face_landmarks_proto,
				connections=mp.solutions.face_mesh.FACEMESH_IRISES,
				landmark_drawing_spec=None,
				connection_drawing_spec=mp.solutions.drawing_styles
				.get_default_face_mesh_iris_connections_style(),
			)

		return annotated_image

	def plot_face_blendshapes_bar_graph(face_blendshapes):
		# Extract the face blendshapes category names and scores
		face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
		face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
		# The blendshapes are ordered in decreasing score value
		face_blendshapes_ranks = range(len(face_blendshapes_names))

		fig, ax = plt.subplots(figsize=(12, 12))
		bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
		ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
		ax.invert_yaxis()

		# Label each bar with values
		for score, patch in zip(face_blendshapes_scores, bar.patches):
			plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

		ax.set_xlabel("Score")
		ax.set_title("Face Blendshapes")
		plt.tight_layout()
		plt.show()

	img = cv2.imread("./business-person.png")
	cv2.imshow("Image", img)

	# Create an FaceLandmarker object
	base_options = python.BaseOptions(model_asset_path="./face_landmarker_v2_with_blendshapes.task")
	options = python.vision.FaceLandmarkerOptions(
		base_options=base_options,
		output_face_blendshapes=True,
		output_facial_transformation_matrixes=True,
		num_faces=1,
	)
	detector = python.vision.FaceLandmarker.create_from_options(options)

	# Load the input image
	image = mp.Image.create_from_file("business-person.png")

	# Detect face landmarks from the input image
	detection_result = detector.detect(image)

	# Process the detection result. In this case, visualize it
	annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
	cv2.imshow("Face Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

	# Visualize the face blendshapes categories using a bar graph
	plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])

	# Print the transformation matrix
	print(detection_result.facial_transformation_matrixes)

# REF [function] >> hand_landmark_detection_test()
def face_detection_test():
	import cv2
	import mediapipe as mp

	detector = mp.solutions.face_detection.FaceDetection(
		min_detection_confidence=0.5,
		model_selection=1,
	)

	video = cv2.VideoCapture(0)
	while video.isOpened():
		ret, img = video.read()
		if not ret:
			continue

		img = cv2.flip(img, 1)
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		detection_result = detector.process(img_rgb)  # Detect faces
		if detection_result.detections is not None:
			#print(detection_result.detections)
			for detection in detection_result.detections:
				mp.solutions.drawing_utils.draw_detection(
					img,
					detection=detection,
					keypoint_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
						color=(0, 255, 0),
						thickness=1,
						circle_radius=2,
					),
					bbox_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
						color=(0, 0, 255),
					),
				)

		key = cv2.waitKey(10)
		if key == 27:
			break

		cv2.imshow("Face Detection", img)

	video.release()
	cv2.destroyAllWindows()

# REF [site] >>
#	https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
#	https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
def pose_landmarks_detection_example():
	# Download:
	#	wget -O pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
	#	wget https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg

	import numpy as np
	import cv2
	import mediapipe as mp
	from mediapipe.tasks import python
	from mediapipe.framework.formats import landmark_pb2

	def draw_landmarks_on_image(rgb_image, detection_result):
		pose_landmarks_list = detection_result.pose_landmarks
		annotated_image = np.copy(rgb_image)

		# Loop through the detected poses to visualize
		for idx in range(len(pose_landmarks_list)):
			pose_landmarks = pose_landmarks_list[idx]

			# Draw the pose landmarks
			pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
			pose_landmarks_proto.landmark.extend([
				landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
			])
			mp.solutions.drawing_utils.draw_landmarks(
				annotated_image,
				pose_landmarks_proto,
				mp.solutions.pose.POSE_CONNECTIONS,
				mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
			)
		return annotated_image

	img = cv2.imread("./girl-4051811_960_720.jpg")
	cv2.imshow("Image", img)

	# Create an PoseLandmarker object
	base_options = python.BaseOptions(model_asset_path="./pose_landmarker.task")
	options = python.vision.PoseLandmarkerOptions(
		base_options=base_options,
		output_segmentation_masks=True,
	)
	detector = python.vision.PoseLandmarker.create_from_options(options)

	# Load the input image
	image = mp.Image.create_from_file("./girl-4051811_960_720.jpg")

	# Detect pose landmarks from the input image
	detection_result = detector.detect(image)

	# Process the detection result. In this case, visualize it
	annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
	cv2.imshow("Pose Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

	# Visualize the pose segmentation mask
	segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
	visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
	cv2.imshow("Pose Segmentation Mask", visualized_mask)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def main():
	# MediaPipe
	#	https://github.com/google-ai-edge/mediapipe
	#	https://ai.google.dev/edge/mediapipe/solutions/examples
	#	https://ai.google.dev/edge/mediapipe/solutions/guide
	#	https://ai.google.dev/edge/mediapipe/framework

	# MediaPipe Model Maker
	#	https://ai.google.dev/edge/mediapipe/solutions/model_maker
	#	A tool for customizing existing machine learning (ML) models to work with your data and applications
	# MediaPipe Studio
	#	https://ai.google.dev/edge/mediapipe/solutions/studio
	#	A web-based application for evaluating and customizing on-device ML models and pipelines for your applications

	# Available solutions
	#	https://ai.google.dev/edge/mediapipe/solutions/guide
	#
	#	Generative AI tasks:
	#		LLM inference
	#		Retrieval augmented generation (RAG)
	#		Function calling
	#		Image generation
	#
	#	Vison tasks:
	#		Object detection
	#		Image classification
	#		Image segmentation
	#		Interactive segmentation
	#		Gesture recognition
	#			https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer
	#			Canned gestures: "None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou" 
	#		Hand landmark detection
	#		Image embedding
	#		Face detection
	#		Face landmark detection
	#		Pose landmark detection
	#		Holistic landmarks detection
	#
	#	Text tasks:
	#		Text classification
	#		Text embedding
	#		Language detection
	#
	#	Audio tasks:
	#		Audio classification

	# Install:
	#	pip install mediapipe

	#--------------------

	#object_detection_example()
	#image_classification_example()
	#image_segmentation_example()

	#--------------------
	# Hand

	#hand_landmarks_detection_example()
	hand_landmark_detection_test()

	#gesture_recognition_example()

	#--------------------
	# Face

	#face_detection_example()
	#face_landmarks_detection_example()

	#face_detection_test()

	#--------------------
	# Human

	#pose_landmarks_detection_example()

#-----------------------------------------------------------

if "__main__" == __name__:
	main()
