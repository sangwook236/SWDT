#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace tbb {

void simple_loop_parallelization();
void complex_loop_parallelization();

}  // namespace tbb

int tbb_main(int argc, char *argv[])
{
	tbb::simple_loop_parallelization();
	//tbb::complex_loop_parallelization();  // not yet implemented

	return 0;
}
