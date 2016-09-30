#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

void transform_1()
{
	// Allocate three device_vectors with 10 elements.
	thrust::device_vector<int> X(10);
	thrust::device_vector<int> Y(10);
	thrust::device_vector<int> Z(10);

	// Initialize X to 0,1,2,3, ....
	thrust::sequence(X.begin(), X.end());

	// Compute Y = -X.
	thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<int>());

	// Fill Z with twos.
	thrust::fill(Z.begin(), Z.end(), 2);

	// Compute Y = X mod 2.
	thrust::transform(X.begin(), X.end(), Z.begin(), Y.begin(), thrust::modulus<int>());

	// Replace all the ones in Y with tens
	thrust::replace(Y.begin(), Y.end(), 1, 10);

	// Print Y.
	thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, ", "));
	std::cout << std::endl;
}

struct saxpy_functor
{
	saxpy_functor(float _a) : a(_a) {}

	__host__ __device__
	float operator()(const float &x, const float &y) const
	{
		return a * x + y;
	}

	const float a;
};

void saxpy_fast(float A, thrust::device_vector<float> &X, thrust::device_vector<float> &Y)
{
	// Y <- A * X + Y.
	thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(float A, thrust::device_vector<float> &X, thrust::device_vector<float> &Y)
{
	thrust::device_vector<float> temp(X.size());

	// temp <- A.
	thrust::fill(temp.begin(), temp.end(), A);
	// temp <- A * X.
	thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());
	// Y <- A * X + Y.
	thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}

void transform_2()
{
	const float A = 2.0f;

	// Allocate three device_vectors with 10 elements.
	thrust::device_vector<float> X(10);
	thrust::device_vector<float> Y(10);

	// Initialize X & Y to 0,1,2,3, ....
	thrust::sequence(X.begin(), X.end());
	thrust::sequence(Y.begin(), Y.end());

	saxpy_fast(A, X, Y);

	// Print Y.
	thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, ", "));
	std::cout << std::endl;

	//
	thrust::sequence(Y.begin(), Y.end());

	saxpy_fast(A, X, Y);

	// Print Y.
	thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, ", "));
	std::cout << std::endl;
}

void reduce_1()
{
	// Allocate three device_vectors with 10 elements.
	thrust::device_vector<int> D(10);

	// Initialize D to 0,1,2,3, ....
	thrust::sequence(D.begin(), D.end());

	const int sum = thrust::reduce(D.begin(), D.end(), 0, thrust::plus<int>());
	std::cout << "sum = " << sum << std::endl;

	// Put three 1s in a device_vector.
	thrust::device_vector<int> dev_vec(5,0);
	dev_vec[1] = 1;
	dev_vec[3] = 1;
	dev_vec[4] = 1;

	// Count the 1s.
	const int count = thrust::count(dev_vec.begin(), dev_vec.end(), 1);
	std::cout << "count = " << count << std::endl;
}

// square<T> computes the square of a number f(x) -> x*x.
template <typename T>
struct square
{
	__host__ __device__
	T operator()(const T &x) const
	{
		return x * x;
	}
};

void reduce_2()
{
	// Initialize host array.
	const float x[4] = { 1.0f, 2.0f, 3.0f, 4.0f };

	// Transfer to device.
	thrust::device_vector<float> dev_vec(x, x + 4);

	// Compute norm.
	const float init = 0;
	const float norm = std::sqrt(thrust::transform_reduce(dev_vec.begin(), dev_vec.end(), square<float>(), init, thrust::plus<float>()));
	std::cout << "norm = " << norm << std::endl;
}

struct arbitrary_functor
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // D[i] = A[i] + B[i] * C[i];
        thrust::get<3>(t) = thrust::get<0>(t) + thrust::get<1>(t) * thrust::get<2>(t);
    }
};

}  // namespace local
}  // unnamed namespace

namespace my_thrust {

void algorithm()
{
	// Transform algorithm --------------------------------
	local::transform_1();
	local::transform_2();

	// Reduce algorithm -----------------------------------
	local::reduce_1();
	local::reduce_2();

	// Scan operation -------------------------------------
	{
		//
		int data1[6] = { 1, 0, 2, 2, 1, 3 };
		int data2[6] = { 1, 0, 2, 2, 1, 3 };

		thrust::inclusive_scan(data1, data1 + 6, data1);  // in-place scan
		thrust::exclusive_scan(data2, data2 + 6, data2);  // in-place scan

		thrust::copy(data1, data1 + 6, std::ostream_iterator<int>(std::cout, ", "));
		std::cout << std::endl;
		thrust::copy(data2, data2 + 6, std::ostream_iterator<int>(std::cout, ", "));
		std::cout << std::endl;
	}

	// Sorting --------------------------------------------
	{
		const int N = 6;

		int A[N] = { 1, 4, 2, 8, 5, 7 };

		thrust::sort(A, A + N);  // A is now { 1, 2, 4, 5, 7, 8 }

		thrust::copy(A, A + N, std::ostream_iterator<int>(std::cout, ", "));
		std::cout << std::endl;

		//
		int keys[N] = { 1, 4, 2, 8, 5, 7 };
		char values[N] = { 'a', 'b', 'c', 'd', 'e', 'f' };

		thrust::sort_by_key(keys, keys + N, values);  // keys is now { 1, 2, 4, 5, 7, 8 }
		thrust::copy(keys, keys + N, std::ostream_iterator<int>(std::cout, ", "));
		std::cout << std::endl;
		thrust::copy(values, values + N, std::ostream_iterator<char>(std::cout, ", "));
		std::cout << std::endl;

		//
		int B[N] = { 1, 4, 2, 8, 5, 7 };

		thrust::stable_sort(B, B + N, thrust::greater<int>());  // B is now { 8, 7, 5, 4, 2, 1 }

		thrust::copy(B, B + N, std::ostream_iterator<int>(std::cout, ", "));
		std::cout << std::endl;
	}

	// for_each -------------------------------------------
	{
		// Allocate storage.
		thrust::device_vector<float> A(5);
		thrust::device_vector<float> B(5);
		thrust::device_vector<float> C(5);
		thrust::device_vector<float> D(5);

		// Initialize input vectors.
		A[0] = 3;  B[0] = 6;  C[0] = 2; 
		A[1] = 4;  B[1] = 7;  C[1] = 5; 
		A[2] = 0;  B[2] = 2;  C[2] = 7; 
		A[3] = 8;  B[3] = 1;  C[3] = 4; 
		A[4] = 2;  B[4] = 8;  C[4] = 3; 

		// Apply the transformation.
		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin(), D.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(A.end(),   B.end(),   C.end(),   D.end())),
			local::arbitrary_functor()
		);

		// Print the output.
		for (int i = 0; i < 5; i++)
			std::cout << A[i] << " + " << B[i] << " * " << C[i] << " = " << D[i] << std::endl;
	}
}

}  // namespace my_thrust
