# Path to libcudnn.so.5.
#export LD_LIBRARY_PATH=/home/sangwooklee/lib_repo/cpp/cuda/lib64:/usr/local/cuda-8.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

from tf_unet import unet, util, image_util
import numpy as np
from PIL import Image
import os

#%%------------------------------------------------------------------
# Load data.

#dataset_home_dir_path = "/home/shared1/sangwook_dataset"
dataset_home_dir_path = "D:/dataset"

dataset_dir_path = dataset_home_dir_path + "/my_dataset/biomedical_imaging/isbi2012_em_segmentation_challenge"

#train_dataset_search_pattern = dataset_dir_path + "/train-volume.tif"
train_dataset_search_pattern = dataset_dir_path + "/*.tif"
test_dataset_search_pattern = dataset_dir_path + "/*.tif"

model_output_dir_path = dataset_dir_path + "/output"
prediction_dir_path = dataset_dir_path + "/prediction"

if not os.path.exists(prediction_dir_path):
	try:
		os.makedirs(prediction_dir_path)
	except OSError as exception:
		if exception.errno != os.errno.EEXIST:
			raise

#%%------------------------------------------------------------------
# Setup model.

net = unet.Unet(layers = 3, features_root = 64, channels = 1, n_class = 2)
#net = unet.Unet(layers = 3, features_root = 32, channels = 1, n_class = 2)
#net = unet.Unet(layers = 3, features_root = 128, channels = 1, n_class = 2)
#net = unet.Unet(layers = 3, features_root = 256, channels = 1, n_class = 2)
#net = unet.Unet(layers = 3, features_root = 512, channels = 1, n_class = 2)
#net = unet.Unet(layers = 3, features_root = 1024, channels = 1, n_class = 2)  # Error.
#net = unet.Unet(layers = 5, features_root = 256, channels = 1, n_class = 2)
#net = unet.Unet(layers = 7, features_root = 256, channels = 1, n_class = 2)

#%%------------------------------------------------------------------
# Train.

#train_data_provider = image_util.ImageDataProvider(train_dataset_search_pattern)
train_data_provider = image_util.ImageDataProvider(search_path = train_dataset_search_pattern, data_suffix = '.tif', mask_suffix = '_mask.tif')

trainer = unet.Trainer(net)

model_filepath = trainer.train(train_data_provider, model_output_dir_path, training_iters = 32, epochs = 100)
#model_filepath = model_output_dir_path + '/model.cpkt'

#%%------------------------------------------------------------------
# Test.

test_data_provider = image_util.ImageDataProvider(search_path = test_dataset_search_pattern, data_suffix = '.tif', mask_suffix = '_mask.tif')

x_tests, y_tests = test_data_provider(30)  # 30 images.

#idx = 3
##x_img = Image.fromarray(np.uint8(x_tests[idx] * 255).reshape(x_tests[idx].shape[0], x_tests[idx].shape[1]), mode='L')
#x_img = Image.fromarray(np.uint8(x_tests[idx,:,:,0] * 255).reshape(x_tests[idx,:,:,0].shape[0], x_tests[idx,:,:,0].shape[1]), mode='L')
#x_img.show()
#y_img = Image.fromarray(np.uint8(y_tests[idx,:,:,0] * 255).reshape(y_tests[idx,:,:,0].shape[0], y_tests[idx,:,:,0].shape[1]))
##y_img = Image.fromarray(np.uint8(y_tests[idx,:,:,1] * 255).reshape(y_tests[idx,:,:,1].shape[0], y_tests[idx,:,:,1].shape[1]))
#y_img.show()

#prediction = net.predict(model_filepath, x_tests);
indexes = range(x_tests.shape[0])
for idx in indexes:
	print("Processing %d-th test image..." % idx)

	x_test = x_tests[idx].reshape(1, x_tests[idx].shape[0], x_tests[idx].shape[1], x_tests[idx].shape[2])
	y_test = y_tests[idx].reshape(1, y_tests[idx].shape[0], y_tests[idx].shape[1], y_tests[idx].shape[2])

	prediction = net.predict(model_filepath, x_test)
	#Image.fromarray(prediction[0,:,:,0], mode='F').show()  % Error: not correct.
	#Image.fromarray(prediction[0,:,:,1], mode='F').show()  % Error: not correct.

	print("Error rate = %f" % unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape)))

	img = util.combine_img_prediction(x_test, y_test, prediction)
	util.save_image(img, prediction_dir_path + "/prediction" + str(idx) + ".jpg")

#%%------------------------------------------------------------------
# Test.

test_data_provider = image_util.ImageDataProvider(dataset_dir_path + "/test_img09.tif")

x_test, y_test = test_data_provider(1)  # Single image.

prediction = net.predict(model_filepath, x_test)

print("Error rate = %f" % unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape)))

img = util.combine_img_prediction(x_test, y_test, prediction)
util.save_image(img, prediction_dir_path + "/prediction.jpg")

#%%------------------------------------------------------------------

import numpy as np

x_test_all, y_test_all = train_data_provider(16)
idx = 3  # 0, 3, 13.

x_test_single = np.zeros((1, 1024, 1000, 1), dtype = np.float64)
y_test_single = np.zeros((1, 1024, 1000, 2), dtype = np.float64)
x_test_single[0] = x_test_all[idx]
y_test_single[0] = y_test_all[idx]

prediction = net.predict(model_filepath, x_test_single)

unet.error_rate(prediction, util.crop_to_shape(y_test_single, prediction.shape))

img = util.combine_img_prediction(x_test_single, y_test_single, prediction)
util.save_image(img, prediction_dir_path + "/prediction.jpg")
