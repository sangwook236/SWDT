//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char* argv[])
{
	int gnuplot_main(int argc, char* argv[]);
	int graphviz_main(int argc, char* argv[]);
	int mathgl_main(int argc, char* argv[]);
	int plplot_main(int argc, char *argv[]);
	int vtk_main(int argc, char* argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "gnuplot -------------------------------------------------------------" << std::endl;
		retval = gnuplot_main(argc, argv);

		std::cout << "\nGraphviz library ----------------------------------------------------" << std::endl;
		//retval = graphviz_main(argc, argv);  // Not yet implemented.

		std::cout << "\nMathGL library ------------------------------------------------------" << std::endl;
		//retval = mathgl_main(argc, argv);
		
		std::cout << "\nPLplot library ------------------------------------------------------" << std::endl;
		//retval = plplot_main(argc, argv);

		std::cout << "\nVTK library ---------------------------------------------------------" << std::endl;
		//retval = vtk_main(argc, argv);  // Not yet implemented.
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
