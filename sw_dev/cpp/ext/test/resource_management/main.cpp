#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int valgrind_main(int argc, char *argv[]);
	int vld_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
#if defined(__linux) || defined(__linux__) || defined(linux) || defined(__unix) || defined(__unix__) || defined(unix)
		std::cout << "Valgrind ------------------------------------------------------------" << std::endl;
		retval = valgrind_main(argc, argv);
#elif defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
		std::cout << "\nVisual Leak Detector (VLD) ------------------------------------------" << std::endl;
		retval = vld_main(argc, argv);
#endif
	}
	catch (const std::bad_alloc &ex)
	{
		std::cerr << "std::bad_alloc caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &ex)
	{
		std::cerr << "std::exception caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cerr << "Unknown exception caught." << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
