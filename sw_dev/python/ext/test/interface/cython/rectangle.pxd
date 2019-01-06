cdef extern from 'rectangle.cpp':
	pass

# Decalre the class with cdef.
cdef extern from 'rectangle.h' namespace 'shapes':
	cdef cppclass Rectangle:
		Rectangle() except +
		Rectangle(int, int, int, int) except +

		int getArea()
		void getSize(int* width, int* height)
		void move(int, int)

		int x0, y0, x1, y1
