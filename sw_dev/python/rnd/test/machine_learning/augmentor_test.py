import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
sys.path.append(swl_python_home_dir_path + '/src')

#%%------------------------------------------------------------------
# Prepare dataset.

from swl.machine_learning.data_preprocessing import standardize_samplewise, standardize_featurewise
import numpy as np

if 'posix' == os.name:
	#dataset_home_dir_path = '/home/sangwook/my_dataset'
	dataset_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	dataset_home_dir_path = 'D:/dataset'

#dataset_dir_path = '../../data/machine_learning/data_only1'
dataset_dir_path = '../../data/machine_learning/data_only2'

image_width, image_height = None, None
#image_width, image_height = 300, 200

#%%------------------------------------------------------------------
# Do data augmentation.

if 'posix' == os.name:
	lib_home_dir_path = '/home/sangwook/lib_repo/python'
else:
	lib_home_dir_path = 'D:/lib_repo/python'
	#lib_home_dir_path = 'D:/lib_repo/python/rnd'
sys.path.append(lib_home_dir_path + '/Augmentor_github')

# REF [site] >> https://github.com/mdbloice/Augmentor

import Augmentor

p = Augmentor.Pipeline(source_directory=dataset_dir_path, output_directory='augmentor_output', save_format='PNG')

p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
#p.rotate90(probability=0.5)
#p.rotate270(probability=0.5)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.crop_random(probability=1.0, percentage_area=0.5)
p.resize(probability=1.0, width=120, height=120)
p.status()

#%%------------------------------------------------------------------
# Generate sample in the specified output directory.

#p.sample(100)

#%%------------------------------------------------------------------
# Create a generator.

batch_size = 5

generator = p.keras_generator(batch_size=batch_size)

#x_train, y_train = load_dataset(dataset_dir_path)
#generator = p.keras_generator_from_array(x_train, y_train, batch_size=batch_size)

#images, labels = next(generator)

batch_idx = 0
for batch_images, batch_labels in generator:
	#print(batch_idx, type(batch_images), type(batch_labels))
	print(batch_idx, ':', batch_images.shape, ',', batch_labels.shape)
	batch_idx += 1
