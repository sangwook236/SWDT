import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
sys.path.append(swl_python_home_dir_path + '/src')

#--------------------
import numpy as np
import pandas as pd
from sklearn import preprocessing
#import keras
#from keras.preprocessing.image import ImageDataGenerator

#--------------------
if 'posix' == os.name:
	#data_home_dir_path = '/home/sangwook/my_dataset'
	data_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	data_home_dir_path = 'D:/dataset'

data_dir_path = data_home_dir_path + '/pattern_recognition/uci/wine'

#%%------------------------------------------------------------------
# Use pandas.DataFrame & sklearn.preprocessing.

# REF [site] >> http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

#df = pd.io.parsers.read_csv(
#		data_dir_path + '/wine.data',
#		header=None)
#df.columns=['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline' ]
df = pd.io.parsers.read_csv(
		data_dir_path + '/wine.data',
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

#%%------------------------------------------------------------------

from sklearn.cross_validation import train_test_split

X_wine = df.values[:,1:]
y_wine = df.values[:,0]
X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.30, random_state=None)

std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

# Dimensionality reduction via Principal Component Analysis (PCA).
from sklearn.decomposition import PCA

# On non-standardized data.
pca = PCA(n_components=2).fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# On standardized data.
pca_std = PCA(n_components=2).fit(X_train_std)
X_train_std = pca_std.transform(X_train_std)
X_test_std = pca_std.transform(X_test_std)

def plot3():
	fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))

	for l,c,m in zip(range(1,4), ('blue', 'red', 'green'), ('^', 's', 'o')):
		ax1.scatter(X_train[y_train==l, 0], X_train[y_train==l, 1], color=c, label='class %s' %l, alpha=0.5, marker=m)

	for l,c,m in zip(range(1,4), ('blue', 'red', 'green'), ('^', 's', 'o')):
		ax2.scatter(X_train_std[y_train==l, 0], X_train_std[y_train==l, 1], color=c, label='class %s' %l, alpha=0.5, marker=m)

	ax1.set_title('Transformed NON-standardized training dataset after PCA')    
	ax2.set_title('Transformed standardized training dataset after PCA')    

	for ax in (ax1, ax2):
		ax.set_xlabel('1st principal component')
		ax.set_ylabel('2nd principal component')
		ax.legend(loc='upper right')
		ax.grid()
	plt.tight_layout()

plot3()
plt.show()

# Training a naive Bayes classifier.
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# On non-standardized data.
gnb = GaussianNB()
fit = gnb.fit(X_train, y_train)

# On standardized data.
gnb_std = GaussianNB()
fit_std = gnb_std.fit(X_train_std, y_train)

pred_train = gnb.predict(X_train)
pred_test = gnb.predict(X_test)

print('Prediction accuracy for the training dataset = {:.2%}'.format(metrics.accuracy_score(y_train, pred_train)))
print('Prediction accuracy for the test dataset = {:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))

pred_train_std = gnb_std.predict(X_train_std)
pred_test_std = gnb_std.predict(X_test_std)

print('Prediction accuracy for the training dataset (std) = {:.2%}'.format(metrics.accuracy_score(y_train, pred_train_std)))
print('Prediction accuracy for the test dataset (std) = {:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))
