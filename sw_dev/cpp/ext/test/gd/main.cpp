#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char* argv[])
{
	void basic_operation();
	void two_link_arm();

	try
	{
		//basic_operation();

		//
		two_link_arm();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred: " << std::endl;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return 0;
}
