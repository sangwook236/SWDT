# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 21:38:46 2017

@author: sangwook
"""

#%%------------------------------------------------------------------
# Toy problem.

from __future__ import division, print_function
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
plt.rcParams['image.cmap'] = 'gist_earth'

from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util

nx = 572
ny = 572

generator = image_gen.GrayScaleDataProvider(nx, ny, cnt=20)

x_test, y_test = generator(4)  % 4 images.

fig, ax = plt.subplots(1,2, sharey=True, figsize=(8,4))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")

#%%
net = unet.Unet(channels=generator.channels, n_class=generator.n_class, layers=3, features_root=16)
trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(generator, "./unet_trained", training_iters=20, epochs=100, display_step=2)

#%%
prediction = net.predict("./unet_trained/model.cpkt", x_test)

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12,8))
for i in range(ax.shape[0]):
    ax[i, 0].imshow(x_test[i,...,0], aspect="auto")
    ax[i, 1].imshow(y_test[i,...,1], aspect="auto")
    mask = prediction[i,...,1] > 0.9
    ax[i, 2].imshow(mask, aspect="auto")
ax[0, 0].set_title("Input")
ax[0, 1].set_title("Ground truth")
ax[0, 2].set_title("Prediction")
fig.tight_layout()
fig.savefig("docs/toy_problem.png")


#%%------------------------------------------------------------------
# Basic usage.

from tf_unet import unet, util, image_util

dataset_home_path = "/home/sangwook/my_dataset/life_science/isbi"
train_dataset_path = dataset_home_path + "/train-volume.tif"
test_dataset_path = dataset_home_path + "/test-volume.tif"
model_output_path = dataset_home_path + "/output"

# Prepare data loading.
train_data_provider = image_util.ImageDataProvider(train_dataset_path)

#%% Setup & train.
net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2)
trainer = unet.Trainer(net)

path = trainer.train(train_data_provider, model_output_path, training_iters=32, epochs=100)

#%% Veriry.

#%% Test.
#test_data_provider = image_util.ImageDataProvider(test_dataset_path)
#x_test, y_test = test_data_provider(1)
x_test, y_test = train_data_provider(1)
prediction = net.predict(path, x_test)

unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))

img = util.combine_img_prediction(x_test, y_test, prediction)
util.save_image(img, dataset_home_path + "/prediction.jpg")
