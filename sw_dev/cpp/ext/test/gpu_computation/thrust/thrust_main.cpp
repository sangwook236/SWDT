#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_thrust {

}  // namespace my_thrust

int thrust_main(int argc, char *argv[])
{
	try
	{
		throw std::runtime_error("not yet implemented");
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

    return 0;
}
