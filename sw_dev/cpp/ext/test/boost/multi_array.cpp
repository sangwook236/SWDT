#include <boost/multi_array.hpp>
#include <iostream>
#include <cstring>


namespace {
namespace local {

void basic_operation_1()
{
	typedef boost::multi_array<double, 3> array_type;
	typedef array_type::index index_type;
	typedef array_type::index_range range_type;

	// create a 3D array that is 2 x 3 x 4
	const int size1 = 2;
	const int size2 = 3;
	const int size3 = 4;

	//-------------------------------------------------------------------------
	// specifying array dimensions
#if 1
	array_type marrA(boost::extents[size1][size2][size3]);
#else
	boost::array<index_type, 3> shape = {{ size1, size2, size3 }};
	array_type marrA(shape);
#endif

	//
	//marrA[ boost::indices[range_type()][range_type()][range_type()] ] = 1.0;  // error !!!
	memset(marrA.data(), 0, sizeof(double) * size1 * size2 * size3);

	// assign values to the elements
	int values = 0;
	for(index_type i = 0; i != size1; ++i)
		for(index_type j = 0; j != size2; ++j)
			for(index_type k = 0; k != size3; ++k)
				marrA[i][j][k] = ++values;

	{
		double *data = marrA.data();
		for (int i = 0; i < size1 * size2 * size3; ++i)
			std::cout << data[i] << ", ";
		std::cout << std::endl;

		data[0] = -1;
		data[1] = -2;
		data[2] = -3;
		data[3] = -4;
		data[4] = -5;
		for(index_type i = 0; i != size1; ++i)
		{
			for(index_type j = 0; j != size2; ++j)
			{
				for(index_type k = 0; k != size3; ++k)
					std::cout << marrA[i][j][k] << ", ";
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	//-------------------------------------------------------------------------
	//

	std::cout << "dimensionality: " << array_type::dimensionality << std::endl;
	std::cout << "number of dimensionality: " << marrA.num_dimensions() << std::endl;

	// the number of values contained in a. It is equivalent to a.shape()[0].
	std::cout << "size: " << marrA.size() << std::endl;
	std::cout << "number of elements: " << marrA.num_elements() << std::endl;
	std::cout << "address of the elements: " << marrA.origin() << std::endl;
	// if all dimensions of the array are 0-indexed and stored in ascending order, this is equivalent to origin().
	std::cout << "address of the elements: " << marrA.data() << std::endl;

	const array_type::size_type *shape = marrA.shape();
	std::cout << "shape: ";
	for (array_type::size_type i = 0; i < array_type::dimensionality; ++i)
		std::cout << shape[i] << ", ";
	std::cout << std::endl;

	const array_type::index *strides = marrA.strides();
	std::cout << "strides: ";
	for (array_type::size_type i = 0; i < array_type::dimensionality; ++i)
		std::cout << strides[i] << ", ";
	std::cout << std::endl;

	const array_type::index *index_bases = marrA.index_bases();
	std::cout << "index bases: ";
	for (array_type::size_type i = 0; i < array_type::dimensionality; ++i)
		std::cout << index_bases[i] << ", ";
	std::cout << std::endl;

	//-------------------------------------------------------------------------
	// accessing elements
#if 1
	marrA[0][0][0] = 3.14;
#else
	boost::array<index_type, 3> idx = {{ 0, 0, 0 }};
	marrA(idx) = 3.14;
#endif
}

void basic_operation_2()
{
	{
		const size_t K = 3, D = 2;

		double arrA[] = {
			0.5, 0.2,  0.3,
			0.2, 0.4,  0.4,
			0.1, 0.45, 0.45
		};
		double arrB[] = {
			0.5,   0.5,
			0.75,  0.25,
			0.25,  0.75
		};

		boost::multi_array_ref<double, 2> A(arrA, boost::extents[K][K]);
		boost::multi_array_ref<double, 2> B(arrB, boost::extents[K][D]);
		//boost::const_multi_array_ref<double, 2> A(arrA, boost::extents[K][K]);
		//boost::const_multi_array_ref<double, 2> B(arrB, boost::extents[K][D]);

		std::cout << "A =" << std::endl;
		for (size_t i = 0; i < K; ++i)
		{
			for (size_t j = 0; j < K; ++j)
				std::cout << A[i][j] << ' ';
			std::cout << std::endl;
		}

		B[0][1] = -0.25;

		std::cout << "B =" << std::endl;
		for (size_t i = 0; i < K; ++i)
		{
			for (size_t j = 0; j < D; ++j)
				std::cout << B[i][j] << ' ';
			std::cout << std::endl;
		}
	}
}

void array_view()
{
	typedef boost::multi_array<double, 3> array_type;
	typedef array_type::index index_type;
	typedef array_type::index_range range_type;

	// create a 3D array that is 2 x 3 x 4
	const int size1 = 2;
	const int size2 = 3;
	const int size3 = 4;

#if 1
	array_type marrA(boost::extents[size1][size2][size3]);
#else
	boost::array<index_type, 3> shape = {{ size1, size2, size3 }};
	array_type marrA(shape);
#endif

	//
	//marrA[ boost::indices[range_type()][range_type()][range_type()] ] = 1.0;  // error !!!
	memset(marrA.data(), 0, sizeof(double) * size1 * size2 * size3);

	// assign values to the elements
	int values = 0;
	for(index_type i = 0; i != size1; ++i)
		for(index_type j = 0; j != size2; ++j)
			for(index_type k = 0; k != size3; ++k)
				marrA[i][j][k] = ++values;

	//-------------------------------------------------------------------------
	// creating views

	// array_view dims: [base,bound) (dimension striding default = 1)
	// dim 0: [0,2)
	// dim 1: [1,3)
	// dim 2: [0,4) (strided by 2)
	array_type::array_view<3>::type myview1 = marrA[ boost::indices[range_type(0,2)][range_type(1,3)][range_type(0,4,2)] ];

	// array_view dims:
	// [base,stride,bound)
	// [0,1,2), 1, [0,2,4)
	array_type::index_gen indices;
	array_type::array_view<2>::type myview2 = marrA[ indices[range_type(0,2)][1][range_type(0,4,2)] ];

	array_type::array_view<3>::type myview3 = marrA[ boost::indices[range_type()][range_type() < 3 ][1 <= range_type().stride(2) <= 3] ];
}

void array_ordering_and_base()
{
	typedef boost::multi_array<double, 3> array_type;

	//-------------------------------------------------------------------------
	// storage ordering

	// create a 3D array that is 2 x 3 x 4
	const int size1 = 2;
	const int size2 = 3;
	const int size3 = 4;

	array_type marrB1(boost::extents[size1][size2][size3], boost::c_storage_order());  // default
	//call_c_function(marrB1.data());
	array_type marrB2(boost::extents[size1][size2][size3], boost::fortran_storage_order());
	//call_fortran_function(marrB2.data());

	//
	typedef boost::general_storage_order<3> storage_type;

	// store last dimension, then first, then middle
	array_type::size_type ordering[] = { 2, 0, 1 };

	// store the first dimension(dimension 0) in descending order
	bool ascending[] = { false, true, true };

	array_type marrB3(boost::extents[3][4][2], storage_type(ordering, ascending));

	//-------------------------------------------------------------------------
	// setting the array base

	//typedef boost::multi_array_types::extent_range extent_range_type;
	typedef array_type::extent_range extent_range_type;

	array_type::extent_gen extents;

	// dimension 0: 0-based
	// dimension 1: 1-based
	// dimension 2: (-1)-based
	array_type marrC1(extents[2][extent_range_type(1,4)][extent_range_type(-1,3)]);

	// to set all bases to the same value
	marrC1.reindex(1);

	// dimension 0: 0-based
	// dimension 1: 1-based
	// dimension 2: (-1)-based
	array_type marrC2(extents[2][3][4]);
	boost::array<array_type::index, 3> bases = {{ 0, 1, -1 }};
	marrC2.reindex(bases);
}

void array_size()
{
	typedef boost::multi_array<double, 3> array_type;

	//-------------------------------------------------------------------------
	// changing an array's shape

	array_type marrD(boost::extents[2][3][4]);
	boost::array<array_type::index, 3> dims = {{ 4, 3, 2 }};
	marrD.reshape(dims);

	//-------------------------------------------------------------------------
	// resizing an array
	array_type marrE(boost::extents[3][4][2]);
	marrE[0][0][0] = 4;
	marrE[2][2][1] = 5;
#if defined(NDEBUG) || defined(_STLPORT_VERSION)
	marrE.resize(extents[2][3][4]);
#else
	// FIXME [modify] >> MSVC: compile-time error in debug build
	//	I don't know why
	//marrE.resize(extents[2][3][4]);
	//marrE = array_type(extents[2][3][4]);
#endif
	std::cout << std::endl;
	assert(marrE[0][0][0] == 4);
	//marrE[2][2][1] is no longer valid
}

}  // namespace local
}  // unnamed namespace

void multi_array()
{
	local::basic_operation_1();
	local::basic_operation_2();
	local::array_view();
	local::array_ordering_and_base();
	local::array_size();
}
