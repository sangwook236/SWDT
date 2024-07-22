#if defined(_WIN64) || defined(_WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


int main(int argc, char *argv[])
{
	int galib_main(int argc, char *argv[]);
	int libcmaes_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "GAlib library -------------------------------------------------------" << std::endl;
		//  - Genetic algorithm.
		//retval = galib_main(argc, argv);

		std::cout << "\nlibcmaes ------------------------------------------------------------" << std::endl;
		//  - Covariance matrix adaptation evolution strategy (CMA-ES).
		retval = libcmaes_main(argc, argv);
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
