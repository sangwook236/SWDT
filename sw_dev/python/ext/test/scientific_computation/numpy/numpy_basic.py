# REF [site] >> https://docs.scipy.org/doc/numpy/user/quickstart.html

import numpy as np

#%%-------------------------------------------------------------------

a = np.arange(30).reshape(3, 5, 2)
#a = np.zeros((3, 5, 2), dtype = np.complex128)

a.ndim
a.size
a.shape

a.dtype
a.dtype.name
a.itemsize
type(a)

np.prod(a.shape)

a = np.array([2, 3, 4])
a.dtype
b = np.array([1.2, 3.5, 5.1])
b.dtype

c = np.array([(1.5,2,3), (4,5,6)])
c
d = np.array([[1,2], [3,4]], dtype = complex)
d

#%%-------------------------------------------------------------------
# Create array.

a = np.ndarray(shape=(4, 2), dtype=np.float, order='F')  # At random.

b = np.zeros((3, 4))
c = np.ones((2, 3, 4), dtype=np.int16)
d = np.empty((2, 3))
e = np.full((2, 3), 6)

f = np.arange(10, 30, 5)
g = np.arange(0, 2, 0.3)

from numpy import pi

#x = np.linspace(0, 2, 9)
x = np.linspace(0, 2 * pi, 100)
f = np.sin(x)

#%%-------------------------------------------------------------------
# Print array.

a = np.arange(6)
print(a)
b = np.arange(12).reshape(4, 3)
print(b)
c = np.arange(24).reshape(2, 3, 4)
print(c)

#%%-------------------------------------------------------------------
# Conversion.

a = np.random.random((2, 3))
np.int32(np.round(a))
a.astype(np.int)

#%%-------------------------------------------------------------------
# Basic operation.

a = np.array([20, 30, 40, 50])
b = np.arange(4)

c = a - b
b**2
10 * np.sin(a)
a < 35

A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 0], [3, 4]])

# Elementwise product.
A * B
# Matrix product.
A.dot(B)
np.dot(A, B)

a = np.ones((2, 3), dtype = int)
b = np.random.random((2, 3))
a *= 3
b += a
# b is not automatically converted to integer type.
#a += b

a = np.ones(3, dtype = np.int32)
b = np.linspace(0, pi, 3)
b.dtype.name
c = a + b
c.dtype.name
d = np.exp(c * 1j)
d.dtype.name

a = np.random.random((2, 3))
a
a.sum()
a.min()
a.max()

b = np.arange(12).reshape(3, 4)
b
# Sum of each column.
b.sum(axis = 0)
# Min of each row.
b.min(axis = 1)
# Cumulative sum along each row.
b.cumsum(axis = 1)

#%%-------------------------------------------------------------------
# Universal function.

B = np.arange(3)
B
np.exp(B)
np.sqrt(B)

C = np.array([2., -1., 4.])
np.add(B, C)

#%%-------------------------------------------------------------------
# Index, slice, and iterate.

#%%-------------------------------------------------------------------
# Shape manipulation.

#%%-------------------------------------------------------------------
# Copy and view.

#%%-------------------------------------------------------------------
# Automatic reshaping.

a = np.arange(30)
a.shape = 2, -1, 3  # -1 means "whatever is needed".
a.shape
a

#%%-------------------------------------------------------------------
# Vector stacking.

x = np.arange(0, 10, 2)
y = np.arange(5)

m = np.vstack([x, y])
xy = np.hstack([x, y])

#--------------------
a = [np.array([[11,12], [13,14], [15,16]]), np.array([[21,22], [23,24], [25,26]]), np.array([[31,32], [33,34], [35,36]]), np.array([[41,42], [43,44], [45,46]])]
b = np.array(a)
c = b.tolist()

np.stack(a, axis=1)
np.stack(b, axis=1)
np.stack(c, axis=1)

#%%-------------------------------------------------------------------
# Handling NaN and Infinity.
#	REF [library] >> pandas for handling NaN.

# np.nan: np.nansum, np.nanmin, np.nanmax, np.nanmean.
# np.inf.

np.array([0, 1, np.nan, np.inf, -10, 10]) * np.nan  # array([nan, nan, nan, nan, nan, nan]).

a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 2, 3, 4, None])
c = np.array([1, 2, 3, 4, np.nan])
d = np.array([1, 2, 3, 4, np.inf])

np.mean(a)  # 3.0.
np.nanmean(a)  # 3.0.

#np.mean(b)  # Error.
#np.nanmean(b)  # Error.

np.mean(c)  # np.nan.
np.nanmean(c)  # 2.5.

np.mean(d)  # np.inf.
np.nanmean(d)  # np.inf.

#%%------------------------------------------------------------------
# Save & load a numpy.array to an npy file.

arr = np.arange(10).reshape(2, 5)
np.save('./arr.npy', arr)

arr_loaded = np.load('./arr.npy')

#----------
x1, y1 = np.arange(10).reshape(2, 5), np.arange(10, 20).reshape(2, 5)
x2, y2 = np.arange(20).reshape(4, 5), np.arange(20, 40).reshape(4, 5)

x_filepaths, y_filepaths = './x_npzfile.npz', './y_npzfile.npz'

np.savez(x_filepaths, x1, x2)
#np.savez(x_filepaths, *(x1, x2))
#np.savez(y_filepaths, y1, y2)
np.savez(y_filepaths, y2, y1)

x_npzfile = np.load(x_filepaths)
y_npzfile = np.load(y_filepaths)
#print(type(x_npzfile), type(x_files.files))

print('X files =', x_npzfile.files)
print('Y files =', y_npzfile.files)
print('X file 0 =', x_npzfile['arr_0'])
print('X file 1 =', x_npzfile['arr_1'])
print('Y file 0 =', y_npzfile['arr_0'])
print('Y file 1 =', y_npzfile['arr_1'])

#----------
np.savez(x_filepaths, x1=x1, x2=x2)
#np.savez(y_filepaths, y1=y1, y2=y2)
np.savez(y_filepaths, y2=y2, y1=y1)

x_npzfile = np.load(x_filepaths)
y_npzfile = np.load(y_filepaths)

print('X files =', x_npzfile.files)
print('Y files =', y_npzfile.files)
print('X file 0 =', x_npzfile['x1'])
print('X file 1 =', x_npzfile['x2'])
print('Y file 0 =', y_npzfile['y1'])
print('Y file 1 =', y_npzfile['y2'])

#for xk, yk in zip(x_npzfile.keys(), y_npzfile.keys()):
for xk, yk in zip(sorted(x_npzfile.keys()), sorted(y_npzfile.keys())):
	print(xk, yk)
