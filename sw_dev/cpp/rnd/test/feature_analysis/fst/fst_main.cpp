#include <error.hpp>
#include <iostream>
#include <string>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_fst {

void demo_10();

}  // namespace my_fst

int fst_main(int argc, char *argv[])
{
	try
	{
		my_fst::demo_10();
	}
	catch (const FST::fst_error &e)
	{
		std::cout << "FST exception caught: " << e.what() << ", code = " << e.code() << std::endl;
		return 1;
	}

	return 0;
}
