#if defined(WIN32) || defined(_WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int dlib_main(int argc, char *argv[]);
	int loki_main(int argc, char *argv[]);
	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "dlib library --------------------------------------------------------" << std::endl;
		//  Algorithm, threading, networking, parsing.
		//  Graphical user interfaces (GUI).
		//  Linear algebra, numerical algorithms.
		//  Optimization.
		//  Image processing.
		//  Machine learning algorithms.
		//  Graph tools, Bayesian networks, graphical model inference algorithms.
		//  Data compression and integrity algorithms.
		//  Testing, general utilities.
		retval = dlib_main(argc, argv);

		std::cout << "\nLoki library --------------------------------------------------------" << std::endl;
		//retval = loki_main(argc, argv);  // Not yet implemented.
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
