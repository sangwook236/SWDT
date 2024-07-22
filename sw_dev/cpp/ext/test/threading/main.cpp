#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int std_thread_main(int argc, char *argv[]);
	int tbb_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "C++ Multithreading --------------------------------------------------" << std::endl;
		retval = std_thread_main(argc, argv);

		std::cout << "\nBoost.Thread libary -------------------------------------------------" << std::endl;
		// REF [library] >> Boost library.

		std::cout << "\nThreading Building Blocks (TBB) library -----------------------------" << std::endl;
		retval = tbb_main(argc, argv);
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
