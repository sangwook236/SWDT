# REF [site] >> http://scikit-learn.org/stable/modules/preprocessing.html

from sklearn import preprocessing
import numpy as np

X = np.array([[1.0, -1.0, 2.0],
	[2.0, 0.0, 0.0],
	[0.0, 1.0, -1.0]])

#%%-------------------------------------------------------------------
# Standardization, or mean removal and variance scaling.

X_scaled = preprocessing.scale(X)
#X_scaled = preprocessing.minmax_scale(X)  # [0, 1].
#X_scaled = preprocessing.maxabs_scale(X)  # [-1, 1].
#X_scaled = preprocessing.robust_scale(X)
X_scaled.mean(axis=0)
X_scaled.std(axis=0)

scaler = preprocessing.StandardScaler().fit(X)
#scaler = preprocessing.MinMaxScaler().fit(X)  # [0, 1].
#scaler = preprocessing.MaxAbsScaler().fit(X)  # [-1, 1].
#scaler = preprocessing.RobustScaler().fit(X)
print(scaler.mean_)
print(scaler.scale_)
scaler.transform(X)
scaler.transform([[-1.0, 1.0, 0.0]])

#%%-------------------------------------------------------------------
# Whitening.

#sklearn.decomposition.PCA
#sklearn.decomposition.RandomizedPCA

#%%-------------------------------------------------------------------
# Normalization.

X_normalized = preprocessing.normalize(X, norm='l2')

normalizer = preprocessing.Normalizer().fit(X)
normalizer.transform(X)
#normalizer = preprocessing.Normalizer().fit_transform(X)

#%%-------------------------------------------------------------------
# Banarization.

banarizer = preprocessing.Banarizer().fit(X)
#banarizer = preprocessing.Banarizer(threshold=1.1)
banarizer.transform(X)

#%%-------------------------------------------------------------------
# Encoding categorical features.

enc = preprocessing.OneHotEncoder()

#%%-------------------------------------------------------------------
# Imputation of missing values.

#%%-------------------------------------------------------------------
# Generating polynomial features.

#%%-------------------------------------------------------------------
# Custom transformation.

