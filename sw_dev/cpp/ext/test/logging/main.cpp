#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char *argv[])
{
	int log4cxx_main(int argc, char *argv[]);
	
	try
	{
		log4cxx_main(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cerr << "exception caught: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cerr << "unknown exception caught ..." << std::endl;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return 0;
}
