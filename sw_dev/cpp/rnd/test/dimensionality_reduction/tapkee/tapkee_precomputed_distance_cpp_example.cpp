#include <tapkee/tapkee.hpp>
#include <tapkee/callbacks/precomputed_callbacks.hpp>
#include <vector>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_tapkee {

// [ref] ${TAPKEE_HOME}/examples/precomputed.cpp
void precomputed_distance_cpp_example()
{
	const int N = 100;
	tapkee::DenseMatrix distances(N, N);
	std::vector<tapkee::IndexType> indices(N);
	for (int i = 0; i < N; ++i)
	{
		indices[i] = i;

		for (int j = 0; j < N; ++j)
			distances(i, j) = std::abs(i - j);
	}

	tapkee::precomputed_distance_callback distance(distances);

	tapkee::TapkeeOutput output = tapkee::initialize()
		.withParameters((tapkee::method = tapkee::MultidimensionalScaling, tapkee::target_dimension = 1))
		.withDistance(distance)
		.embedUsing(indices);

	std::cout << output.embedding.transpose() << std::endl;
}

}  // namespace my_tapkee
