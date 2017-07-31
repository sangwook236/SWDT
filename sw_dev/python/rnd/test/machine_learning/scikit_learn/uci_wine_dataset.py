import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
sys.path.append(swl_python_home_dir_path + '/src')

#%%------------------------------------------------------------------

if 'posix' == os.name:
	#dataset_home_dir_path = '/home/sangwook/my_dataset'
	dataset_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	dataset_home_dir_path = 'D:/dataset'

dataset_dir_path = dataset_home_dir_path + '/pattern_recognition/uci/wine'

import numpy as np
import pandas as pd
from sklearn import preprocessing
#import keras
#from keras.preprocessing.image import ImageDataGenerator

#%%------------------------------------------------------------------
# Use pandas.DataFrame & sklearn.preprocessing.

# REF [site] >> http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

#df = pd.io.parsers.read_csv(
#		dataset_dir_path + '/wine.data',
#		header=None)
#df.columns=['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline' ]
df = pd.io.parsers.read_csv(
		dataset_dir_path + '/wine.data',
		header=None,
		usecols=[0,1,2])
df.columns=['Class label', 'Alcohol', 'Malic acid' ]
df.head()

std_scale = preprocessing.StandardScaler().fit(df[['Alcohol', 'Malic acid']])
df_std = std_scale.transform(df[['Alcohol', 'Malic acid']])

minmax_scale = preprocessing.MinMaxScaler().fit(df[['Alcohol', 'Malic acid']])
df_minmax = minmax_scale.transform(df[['Alcohol', 'Malic acid']])

print('Method 1:')
print('\tMean after standardization:\n\tAlcohol={:.2f}, Malic acid={:.2f}'.format(df_std[:,0].mean(), df_std[:,1].mean()))
print('\tStandard deviation after standardization:\n\tAlcohol={:.2f}, Malic acid={:.2f}'.format(df_std[:,0].std(), df_std[:,1].std()))

print('\tMin-value after min-max scaling:\n\tAlcohol={:.2f}, Malic acid={:.2f}'.format(df_minmax[:,0].min(), df_minmax[:,1].min()))
print('\tMax-value after min-max scaling:\n\tAlcohol={:.2f}, Malic acid={:.2f}'.format(df_minmax[:,0].max(), df_minmax[:,1].max()))

#%%------------------------------------------------------------------
# Plot.

import matplotlib.pyplot as plt

def plot1():
	plt.figure(figsize=(8,6))

	plt.scatter(df['Alcohol'], df['Malic acid'], color='green', label='input scale', alpha=0.5)
	plt.scatter(df_std[:,0], df_std[:,1], color='red', label='Standardized [mu=0, sigma=1]', alpha=0.3)
	plt.scatter(df_minmax[:,0], df_minmax[:,1], color='blue', label='min-max scaled [min=0, max=1]', alpha=0.3)

	plt.title('Alcohol and Malic Acid content of the wine dataset')
	plt.xlabel('Alcohol')
	plt.ylabel('Malic Acid')
	plt.legend(loc='upper left')
	plt.grid()

	plt.tight_layout()

plot1()
plt.show()

def plot2():
	fig, ax = plt.subplots(3, figsize=(6,14))
	for a,d,l in zip(range(len(ax)),
			(df[['Alcohol', 'Malic acid']].values, df_std, df_minmax),
			('Input scale', 'Standardized [mu=0, sigma=1]', 'min-max scaled [min=0, max=1]')):
		for i,c in zip(range(1,4), ('red', 'blue', 'green')):
			ax[a].scatter(d[df['Class label'].values == i, 0], d[df['Class label'].values == i, 1], alpha=0.5, color=c, label='Class %s' %i)
		ax[a].set_title(l)
		ax[a].set_xlabel('Alcohol')
		ax[a].set_ylabel('Malic Acid')
		ax[a].legend(loc='upper left')
		ax[a].grid()

		plt.tight_layout()

plot2()
plt.show()

#%%------------------------------------------------------------------
# Use numpy.array & sklearn.preprocessing.

dataset = df.as_matrix()

dataset_std_scale = preprocessing.StandardScaler().fit(dataset[:,1:])
dataset_std = dataset_std_scale.transform(dataset[:,1:])

dataset_minmax_scale = preprocessing.MinMaxScaler().fit(dataset[:,1:])
dataset_minmax = dataset_minmax_scale.transform(dataset[:,1:])

print('Method 2:')
print('\tMean after standardization:\n\tAlcohol={:.2f}, Malic acid={:.2f}'.format(dataset_std[:,0].mean(), dataset_std[:,1].mean()))
print('\tStandard deviation after standardization:\n\tAlcohol={:.2f}, Malic acid={:.2f}'.format(dataset_std[:,0].std(), dataset_std[:,1].std()))

print('\tMin-value after min-max scaling:\n\tAlcohol={:.2f}, Malic acid={:.2f}'.format(dataset_minmax[:,0].min(), dataset_minmax[:,1].min()))
print('\tMax-value after min-max scaling:\n\tAlcohol={:.2f}, Malic acid={:.2f}'.format(dataset_minmax[:,0].max(), dataset_minmax[:,1].max()))

plot1()
plt.show()

#%%------------------------------------------------------------------
# Use numpy.array & low-level APIs.

dataset_mean = np.mean(dataset[:,1:], axis=0)
dataset_sd = np.std(dataset[:,1:], axis=0)
dataset_std2 = (dataset[:,1:] - dataset_mean) / dataset_sd

dataset_max = np.max(dataset[:,1:], axis=0)
dataset_min = np.min(dataset[:,1:], axis=0)
dataset_minmax2 = (dataset[:,1:] - dataset_min) / (dataset_max - dataset_min)

print('Method 3:')
print('\tMean after standardization:\n\tAlcohol={:.2f}, Malic acid={:.2f}'.format(dataset_std2[:,0].mean(), dataset_std2[:,1].mean()))
print('\tStandard deviation after standardization:\n\tAlcohol={:.2f}, Malic acid={:.2f}'.format(dataset_std2[:,0].std(), dataset_std2[:,1].std()))

print('\tMin-value after min-max scaling:\n\tAlcohol={:.2f}, Malic acid={:.2f}'.format(dataset_minmax2[:,0].min(), dataset_minmax2[:,1].min()))
print('\tMax-value after min-max scaling:\n\tAlcohol={:.2f}, Malic acid={:.2f}'.format(dataset_minmax2[:,0].max(), dataset_minmax2[:,1].max()))

plot1()
plt.show()
