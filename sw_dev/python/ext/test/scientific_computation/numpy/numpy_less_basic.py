# REF [site] >> https://docs.scipy.org/doc/numpy/user/quickstart.html#less-basic

import numpy as np

#%%-------------------------------------------------------------------
# Fancy indexing and index trick.

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

#%%-------------------------------------------------------------------
# ix_() function.

# The ix_ function can be used to combine different vectors so as to obtain the result for each n-uplet.
# For example, if you want to compute all the a+b*c for all the triplets taken from each of the vectors a, b and c.

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
