//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char* argv[])
{
	int graphviz_main(int argc, char* argv[]);
	int mathgl_main(int argc, char* argv[]);
	int plplot_main(int argc, char *argv[]);
	int vtk_main(int argc, char* argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "Graphviz library ----------------------------------------------------" << std::endl;
		//retval = graphviz_main(argc, argv);  // not yet implemented.

		std::cout << "\nMathGL library ------------------------------------------------------" << std::endl;
		retval = mathgl_main(argc, argv);
		
		std::cout << "\nPLplot library ------------------------------------------------------" << std::endl;
		//retval = plplot_main(argc, argv);

		std::cout << "\nVTK library ---------------------------------------------------------" << std::endl;
		//retval = vtk_main(argc, argv);  // not yet implemented.
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
