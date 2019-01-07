# distutils: language=c++

cdef extern from 'arithmetic.cpp':
	pass

cdef extern from 'arithmetic.h' namespace 'arithmetic':
	cpdef double add(const double lhs, const double rhs)
	cpdef double sub(const double lhs, const double rhs)
	cpdef double mul(const double lhs, const double rhs)
	cpdef double div(const double lhs, const double rhs)
