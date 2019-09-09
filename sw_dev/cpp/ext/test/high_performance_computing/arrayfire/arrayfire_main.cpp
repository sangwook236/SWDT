#include <arrayfire.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_arrayfire {

void morphology();

}  // namespace my_arrayfire

int arrayfire_main(int argc, char *argv[])
{
	try
	{
		my_arrayfire::morphology();
	}
	catch (const af::exception &ex)
	{
		std::cout << "af::exception caught: " << ex.what() << std::endl;

		return 1;
	}

	return 0;
}
