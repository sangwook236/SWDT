#include <iostream>
#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opengm {

void markov_chain_example();

}  // namespace my_opengm

int opengm_main(int argc, char *argv[])
{
	my_opengm::markov_chain_example();

	return 0;
}
