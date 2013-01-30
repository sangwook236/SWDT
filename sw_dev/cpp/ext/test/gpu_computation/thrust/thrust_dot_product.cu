#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <iostream>


namespace {
namespace local {

typedef thrust::tuple<float, float, float> Float3;

// this functor implements the dot product between 3d vectors
struct dot_product : public thrust::binary_function<Float3, Float3, float>
{
    __host__ __device__
    float operator()(const Float3 &a, const Float3 &b) const
    {
        return thrust::get<0>(a) * thrust::get<0>(b) +  // x components
			thrust::get<1>(a) * thrust::get<1>(b) +  // y components
            thrust::get<2>(a) * thrust::get<2>(b);  // z components
    }
};

// return a host vector with random values in the range [0,1)
thrust::host_vector<float> random_vector(const size_t N)
{
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

	thrust::host_vector<float> temp(N);
    for (std::size_t i = 0; i < N; ++i)
        temp[i] = u01(rng);

    return temp;
}

}  // namespace local
}  // unnamed namespace

namespace my_thrust {

void dot_product()
{
	const size_t N = 1000;

	thrust::device_vector<float> Ax = local::random_vector(N);  // x components of the 'A' vectors
	thrust::device_vector<float> Ay = local::random_vector(N);  // y components of the 'A' vectors 
	thrust::device_vector<float> Az = local::random_vector(N);  // z components of the 'A' vectors

	thrust::device_vector<float> Bx = local::random_vector(N);  // x components of the 'B' vectors
	thrust::device_vector<float> By = local::random_vector(N);  // y components of the 'B' vectors
	thrust::device_vector<float> Bz = local::random_vector(N);  // z components of the 'B' vectors

	// storage for result of each dot product
	thrust::device_vector<float> result(N);

	// defining a zip_iterator type can be a little cumbersome ...
	typedef thrust::device_vector<float>::iterator                     FloatIterator;
	typedef thrust::tuple<FloatIterator, FloatIterator, FloatIterator> FloatIteratorTuple;
	typedef thrust::zip_iterator<FloatIteratorTuple>                   Float3Iterator;

	// now we'll create some zip_iterators for A and B
	Float3Iterator A_first = thrust::make_zip_iterator(thrust::make_tuple(Ax.begin(), Ay.begin(), Az.begin()));
	Float3Iterator A_last  = thrust::make_zip_iterator(thrust::make_tuple(Ax.end(),   Ay.end(),   Az.end()));
	Float3Iterator B_first = thrust::make_zip_iterator(thrust::make_tuple(Bx.begin(), By.begin(), Bz.begin()));

#if 1
	// finally, we pass the zip_iterators into transform() as if they were 'normal' iterators for a device_vector<local::Float3>.
	thrust::transform(A_first, A_last, B_first, result.begin(), local::dot_product());
#else
	// alternatively, we can avoid creating variables for X_first, X_last, and Y_first and invoke transform() directly.
	thrust::transform(
		thrust::make_zip_iterator(thrust::make_tuple(Ax.begin(), Ay.begin(), Az.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(Ax.end(),   Ay.end(),   Az.end())),
		thrust::make_zip_iterator(thrust::make_tuple(Bx.begin(), By.begin(), Bz.begin())),
		result.begin(),
		local::dot_product()
	);
#endif

	// example output
	//	(0.840188,0.45724,0.0860517) * (0.0587587,0.456151,0.322409) = 0.285683
	//	(0.394383,0.640368,0.180886) * (0.0138811,0.24875,0.0221609) = 0.168775
	//	(0.783099,0.717092,0.426423) * (0.622212,0.0699601,0.234811) = 0.63755
	//	(0.79844,0.460067,0.0470658) * (0.0391351,0.742097,0.354747) = 0.389358
	std::cout << std::fixed;
	for (std::size_t i = 0; i < 4; ++i)
	{
		local::Float3 a = A_first[i];
		local::Float3 b = B_first[i];
		const float dot = result[i];

		std::cout << "(" << thrust::get<0>(a) << "," << thrust::get<1>(a) << "," << thrust::get<2>(a) << ")";
		std::cout << " * ";
		std::cout << "(" << thrust::get<0>(b) << "," << thrust::get<1>(b) << "," << thrust::get<2>(b) << ")";
		std::cout << " = ";
		std::cout << dot << std::endl;
	}
}

}  // namespace my_thrust
