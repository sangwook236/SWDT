#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_thrust {

void iterator()
{
	// constant iterator ----------------------------------
	{
		thrust::constant_iterator<int> first(10);
		thrust::constant_iterator<int> last = first + 3;

		std::cout << first[0] << ", " << first[1] << ", " << first[100] << std::endl;  // 10, 10, 10

		// sum of [first, last)
		std::cout << "sum = " << thrust::reduce(first, last) << std::endl;  // 30 = 10 + 10 + 10
	}

	// counting iterator ----------------------------------
	{
		thrust::counting_iterator<int> first(10);
		thrust::counting_iterator<int> last = first + 3;

		std::cout << first[0] << ", " << first[1] << ", " << first[100] << std::endl;  // 10, 11, 110,

		// sum of [first, last)
		std::cout << "sum = " << thrust::reduce(first, last) << std::endl;  // 33 = 10 + 11 + 12
	}

	// transform iterator ---------------------------------
	{
		thrust::device_vector<int> dev_vec(3);
		dev_vec[0] = 10;
		dev_vec[1] = 20;
		dev_vec[2] = 30;

		thrust::transform_iterator<thrust::negate<int>, thrust::device_vector<int>::iterator> first = thrust::make_transform_iterator(dev_vec.begin(), thrust::negate<int>());
		thrust::transform_iterator<thrust::negate<int>, thrust::device_vector<int>::iterator> last = thrust::make_transform_iterator(dev_vec.end(), thrust::negate<int>());

		std::cout << first[0] << ", " << first[1] << ", " << first[2] << std::endl;  // -10, -20, -30

		// sum of [first, last)
#if 1
		const int sum = thrust::reduce(first, last);
#else
		const int sum = thrust::reduce(
			thrust::make_transform_iterator(dev_vec.begin(), thrust::negate<int>()),
			thrust::make_transform_iterator(dev_vec.end(), thrust::negate<int>())
		);
#endif
		std::cout << "sum = " << sum << std::endl;  // -60 = -10 + -20 + -30
	}

	// permutation iterator -------------------------------
	{
		// gather locations
		thrust::device_vector<int> indexer(4);
		indexer[0] = 3;
		indexer[1] = 1;
		indexer[2] = 0;
		indexer[3] = 5;

		// array to gather from
		thrust::device_vector<int> source(6);
		source[0] = 10;
		source[1] = 20;
		source[2] = 30;
		source[3] = 40;
		source[4] = 50;
		source[5] = 60;

		thrust::permutation_iterator<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> first = thrust::make_permutation_iterator(source.begin(), indexer.begin());
		thrust::permutation_iterator<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> last = thrust::make_permutation_iterator(source.begin(), indexer.end());

		std::cout << first[0] << ", " << first[1] << ", " << first[2] << ", " << first[3] << std::endl;  // 40, 20, 10, 60

		// fuse gather with reduction: sum = source[indexer[0]] + source[indexer[1]] + ...
#if 1
		const int sum = thrust::reduce(first, last);
#else
		const int sum = thrust::reduce(
			thrust::make_permutation_iterator(source.begin(), indexer.begin()),
			thrust::make_permutation_iterator(source.begin(), indexer.end())
		);
#endif
		std::cout << "sum = " << sum << std::endl;  // 130 = 40 + 20 + 10 + 60
	}

	// zip iterator ---------------------------------------
	{
		// initialize vectors
		thrust::device_vector<int> A(3);
		thrust::device_vector<char> B(3);

		A[0] = 10;  A[1] = 20;  A[2] = 30;
		B[0] = 'x';  B[1] = 'y';  B[2] = 'z';

		// create iterator
		thrust::zip_iterator<thrust::tuple<thrust::device_vector<int>::iterator, thrust::device_vector<char>::iterator> > first = thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin()));
		thrust::zip_iterator<thrust::tuple<thrust::device_vector<int>::iterator, thrust::device_vector<char>::iterator> > last = thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end()));

		std::cout << "(" << thrust::get<0>(first[0]) << ", " << thrust::get<1>(first[0]) << "), ";  // returns tuple(10, 'x')
		std::cout << "(" << thrust::get<0>(first[1]) << ", " << thrust::get<1>(first[1]) << "), ";  // returns tuple(20, 'y')
		std::cout << "(" << thrust::get<0>(first[2]) << ", " << thrust::get<1>(first[2]) << ")" << std::endl;  // returns tuple(30, 'z')

		// maximum of [first, last)
		const thrust::tuple<int, char> init = first[0];
		thrust::tuple<int, char> result = thrust::reduce(first, last, init, thrust::maximum<thrust::tuple<int, char> >());
		std::cout << "(" << thrust::get<0>(result) << ", " << thrust::get<1>(result) << ")" << std::endl;  // returns tuple(30, 'z')
	}
}

}  // namespace my_thrust
