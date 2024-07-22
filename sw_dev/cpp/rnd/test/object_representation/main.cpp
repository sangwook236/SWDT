//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


// For using MATLAB library.
//  - MATLAB library path가 LD library path가 설정되어 있어야 함 (for SURFGPU only).
//      e.g.)
//          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/MATLAB/R2012b/bin/glnxa64
//          export LD_LIBRARY_PATH+=/usr/local/MATLAB/R2012b/bin/glnxa64
//          export LD_LIBRARY_PATH+=:/usr/local/MATLAB/R2012b/bin/glnxa64

int main(int argc, char *argv[])
{
	int pictorial_structures_revisited_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "Pictorial structures ------------------------------------------------" << std::endl;
		retval = pictorial_structures_revisited_main(argc, argv);
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
