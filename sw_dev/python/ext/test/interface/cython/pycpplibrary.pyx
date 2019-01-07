# distutils: language = c++

cdef extern from '<cmath>' namespace 'std':
	cpdef double cos(double val)

#cdef extern from '<string>' namespace 'std':
#	string to_string(int val)
