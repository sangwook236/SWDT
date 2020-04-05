#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://docs.scipy.org/doc/numpy/user/quickstart.html

import timeit
import numpy as np

def basic_operation():
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

	#--------------------
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

	#--------------------
	# Print array.

	a = np.arange(6)
	print(a)
	b = np.arange(12).reshape(4, 3)
	print(b)
	c = np.arange(24).reshape(2, 3, 4)
	print(c)

	#--------------------
	# Conversion.

	a = np.random.random((2, 3))
	np.int32(np.round(a))
	a.astype(np.int)

	#--------------------
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

	#--------------------
	# Universal function.

	B = np.arange(3)
	B
	np.exp(B)
	np.sqrt(B)

	C = np.array([2., -1., 4.])
	np.add(B, C)

	#--------------------
	# Index, slice, and iterate.

	#--------------------
	# Shape manipulation.

	#--------------------
	# Copy and view.

	#--------------------
	# Automatic reshaping.

	a = np.arange(30)
	a.shape = 2, -1, 3  # -1 means "whatever is needed".
	a.shape
	a

	#--------------------
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

def handle_NaN_and_infinity()
	# REF [library] >> pandas for handling NaN.

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

def save_and_load_numpy_array_to_npy_file():
	arr = np.arange(10).reshape(2, 5)
	np.save('./arr.npy', arr)

	arr_loaded = np.load('./arr.npy')

	#--------------------
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

	#--------------------
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

def fancy_indexing_and_index_trick():
	a = np.arange(12)**2
	# An array of indices.
	i = np.array([1, 1, 3, 8, 5])
	# The elements of a at the positions i.
	a[i]
	# A bidimensional array of indices.
	j = np.array([[3, 4], [9, 7]])
	# The same shape as j.
	a[j]

	a = np.arange(12).reshape(3,4)
	# Indices for the first dim of a.
	i = np.array([[0, 1], [1, 2]])
	# Indices for the second dim.
	j = np.array([[2, 1], [3, 3]])
	# i and j must have equal shape.
	a[i, j]
	a[i,2]
	a[:,j]
	l = [i, j]
	a[l]  # a[i, j].

	s = np.array([i, j])
	#a[s]  # Error.
	a[tuple(s)]  # a[i, j].

	time = np.linspace(20, 145, 5)
	data = np.sin(np.arange(20)).reshape(5, 4)
	time
	data

	# Index of the maxima for each series.
	ind = data.argmax(axis = 0)
	ind
	# Times corresponding to the maxima.
	time_max = time[ind]
	data_max = data[ind, range(data.shape[1])]  # => data[ind[0],0], data[ind[1],1]...
	time_max
	data_max
	np.all(data_max == data.max(axis = 0))

	a = np.arange(5)
	a[[1, 3, 4]] = 0
	a

	a = np.arange(5)
	a[[0, 0, 2]] = [1, 2, 3]
	a

	a = np.arange(5)
	a[[0,0,2]] += 1
	a

	a = np.arange(12).reshape(3,4)
	b = a > 4
	# b is a boolean with a's shape.
	b
	# 1d array with the selected elements.
	a[b]
	a[b] = 0

	a = np.arange(12).reshape(3, 4)
	b1 = np.array([False, True, True])  # First dim selection.
	b2 = np.array([True, False, True, False])  # Second dim selection.

	a[b1,:]  # Select rows.
	a[b1]  # Same thing.
	a[:,b2]  # Select columns.
	a[b1,b2]  # A weird thing to do.

def ix_function():
	# The ix_ function can be used to combine different vectors so as to obtain the result for each n-uplet.
	# For example, if you want to compute all the a+b*c for all the triplets taken from each of the vectors a, b and c.

	# REF [function] >> np.where().

	a = np.array([2, 3, 4, 5])
	b = np.array([8, 5, 4])
	c = np.array([5, 4, 6, 8, 3])
	ax, bx, cx = np.ix_(a, b, c)
	ax.shape, bx.shape, cx.shape
	result = ax + bx * cx
	result

	def ufunc_reduce(ufct, *vectors):
		vs = np.ix_(*vectors)
		r = ufct.identity
		for v in vs:
			r = ufct(r, v)
		return r

	ufunc_reduce(np.add, a, b, c)

def shuffle_speed_test():
	m = np.random.rand(1000, 50, 60)

	batch_size = 30
	num_repetition = 1000

	# Shuffle an array by shuffling indices.
	def shuffle_by_index(m, batch_size):
		num_elements = len(m)
		indices = np.arange(num_elements)
		np.random.shuffle(indices)
		start_idx = 0
		while start_idx < num_elements:
			end_idx = start_idx + batch_size
			batch_indices = indices[start_idx:end_idx]
			batch = m[batch_indices]

			start_idx += batch_size

	print('Elapsed time =', timeit.timeit(lambda: shuffle_by_index(m, batch_size), number=num_repetition))

	# Shuffle an array itself.
	def suffle_itself(m, batch_size):
		num_elements = len(m)
		np.random.shuffle(m)
		start_idx = 0
		while start_idx < num_elements:
			end_idx = start_idx + batch_size
			batch = m[start_idx:end_idx]

			start_idx += batch_size

	print('Elapsed time =', timeit.timeit(lambda: suffle_itself(m, batch_size), number=num_repetition))

# REF [site] >>
#	https://docs.scipy.org/doc/numpy/reference/maskedarray.html
#	https://docs.scipy.org/doc/numpy/reference/maskedarray.generic.html
def mask_array_test():
	x = np.array([1, 2, 3, -1, 5])

	# Mark the fourth entry as invalid.
	x_masked = np.ma.masked_array(x, mask=[0, 0, 0, 1, 0])

	print('x.mean() =', x.mean())
	print('x_masked.mean() =', x_masked.mean())

def main():
	basic_operation()
	#handle_NaN_and_infinity()
	#save_and_load_numpy_array_to_npy_file()

	#fancy_indexing_and_index_trick()
	#ix_function()

	#shuffle_speed_test()
	
	mask_array_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
