#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <stdexcept>


int main(int argc, char *argv[])
{
	int retval = EXIT_SUCCESS;
	try
	{
		// algorithm, threading, networking, parsing.
		// graphical user interfaces (GUI).
		// linear algebra, numerical algorithms.
		// optimization.
		// image processing.
		// machine learning algorithms.
		// graph tools, Bayesian networks, graphical model inference algorithms.
		// data compression and integrity algorithms.
		// testing, general utilities.

		throw std::runtime_error("not yet implemented");
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
