from libc.math cimport sin

cpdef double f(double x):
	return sin(x * x)

cdef extern from '<math.h>':
	cpdef double cos(double x)

#cdef extern from '<string.h>':
#	char* strstr(const char *haystack, const char *needle)
