#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_tbb {

void simple_loop_parallelization();
void complex_loop_parallelization();

}  // namespace my_tbb

int tbb_main(int argc, char *argv[])
{
	my_tbb::simple_loop_parallelization();
	//my_tbb::complex_loop_parallelization();  // not yet implemented.

	return 0;
}
