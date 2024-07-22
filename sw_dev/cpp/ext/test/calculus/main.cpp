#if defined(_WIN64) || defined(_WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


int main(int argc, char *argv[])
{
	int cppad_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "Boost.Numeric.Odeint library ----------------------------------------" << std::endl;
		// REF [library] >> Boost library.

		std::cout << "\nGSL library -------------------------------------------------------" << std::endl;
		//	- Numerical differentiation.
		//	- Numerical integration.
		//	- Monte Carlo integration.
		// REF [library] >> GSL library.

		std::cout << "\nCOIN-OR CppAD library -----------------------------------------------" << std::endl;
		//  - Automatic differentiation (AutoDiff).
		retval = cppad_main(argc, argv);
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
