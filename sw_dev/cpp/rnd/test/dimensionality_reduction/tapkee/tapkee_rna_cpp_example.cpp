#include <tapkee/tapkee.hpp>
#include <tapkee/callbacks/dummy_callbacks.hpp>
#include <iostream>
#include <fstream>
#include <numeric>
#include <functional>
#include <string>


namespace {
namespace local {

struct MatchKernelCallback
{
	tapkee::ScalarType kernel(const std::string &l, const std::string &r)
	{
		return std::inner_product(l.begin(), l.end(), r.begin(), 0, std::plus<int>(), std::equal_to<std::string::value_type>());
	}
};

}  // namespace local
}  // unnamed namespace

namespace my_tapkee {

// REF [file] >> ${TAPKEE_HOME}/examples/rna.cpp
void RNA_cpp_example()
{
	const std::string input_filename("./data/dimensionality_reduction/tapkee/rna.dat");

	std::ifstream input_stream;
#if defined(__GNUC__)
	input_stream.open(input_filename.c_str());
#else
	input_stream.open(input_filename);
#endif

	std::vector<std::string> rnas;
	std::string line;
	while (!input_stream.eof())
	{
		input_stream >> line;
		rnas.push_back(line);
	}

	local::MatchKernelCallback kernel;

	tapkee::TapkeeOutput result = tapkee::initialize()
		.withParameters((tapkee::method = tapkee::KernelLocallyLinearEmbedding, tapkee::num_neighbors = 30))
		.withKernel(kernel)
		.embedUsing(rnas);

	std::cout << result.embedding.transpose() << std::endl;
}

}  // namespace my_tapkee
