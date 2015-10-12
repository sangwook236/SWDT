#if defined(WIN32)
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
		std::cout << "Dlib library --------------------------------------------------------" << std::endl;
		//  algorithm, threading, networking, parsing.
		//  graphical user interfaces (GUI).
		//  linear algebra, numerical algorithms.
		//  optimization.
		//  image processing.
		//  machine learning algorithms.
		//  graph tools, Bayesian networks, graphical model inference algorithms.
		//  data compression and integrity algorithms.
		//  testing, general utilities.
		//retval = dlib_main(argc, argv);  // not yet implemented.

		std::cout << "\nLoki library --------------------------------------------------------" << std::endl;
		//retval = loki_main(argc, argv);  // not yet implemented.
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
		std::cout << "unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
