//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


// for running SiftGPU & SURFGPU
//  -. nvcc compiler가 path에 설정되어 있어야 함.
//      e.g.) export PATH=$PATH:/usr/local/cuda/bin
//  -. MATlAB library path가 LD library path가 설정되어 있어야 함. (for SURFGPU only)
//      e.g.)
//          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/MATLAB/R2012b/bin/glnxa64
//          export LD_LIBRARY_PATH+=/usr/local/MATLAB/R2012b/bin/glnxa64
//          export LD_LIBRARY_PATH+=:/usr/local/MATLAB/R2012b/bin/glnxa64

int main(int argc, char *argv[])
{
	int lsd_main(int argc, char *argv[]);
	int elsd_main(int argc, char *argv[]);
	int opensift_main(int argc, char *argv[]);
	int siftgpu_main(int argc, char *argv[]);
	int opensurf_main(int argc, char *argv[]);
	int surfgpu_main(int argc, char *argv[]);
	int pictorial_structures_revisited_main(int argc, char *argv[]);
	int fst_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		// line feature ---------------------------------------------
		//retval = lsd_main(argc, argv);

		// ellipse & line feature -----------------------------------
		//retval = elsd_main(argc, argv);

		// local descriptor -----------------------------------------
		//retval = opensift_main(argc, argv);
		//retval = siftgpu_main(argc, argv);

		//retval = opensurf_main(argc, argv);
		//retval = surfgpu_main(argc, argv);

		// pictorial structures -------------------------------------
		//retval = pictorial_structures_revisited_main(argc, argv);

		// Feature Selection Toolbox (FST) library ------------------
		retval = fst_main(argc, argv);
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
