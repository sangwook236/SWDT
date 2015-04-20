#include <tapkee/tapkee.hpp>
#include <iostream>
#include <vector>
#include <cmath>


namespace {
namespace local {

	struct MyDistanceCallback
	{
		tapkee::ScalarType distance(tapkee::IndexType l, tapkee::IndexType r)
		{
			return std::abs(l - r);
		}
	};

}  // namespace local
}  // unnamed namespace

namespace my_tapkee {

	// [ref] ${TAPKEE_HOME}/examples/minimal.cpp
	void minimal_cpp_example()
	{
		const int N = 100;
		std::vector<tapkee::IndexType> indices(N);
		for (int i = 0; i < N; ++i) indices[i] = i;

		local::MyDistanceCallback distance;

		tapkee::TapkeeOutput output = tapkee::initialize()
			.withParameters((tapkee::method = tapkee::MultidimensionalScaling, tapkee::target_dimension = 1))
			.withDistance(distance)
			.embedUsing(indices);

		std::cout << output.embedding.transpose() << std::endl;
	}

}  // namespace my_tapkee
