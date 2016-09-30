//#include "stdafx.h"
#if defined(WIN32) || defined(_WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
    int vexcl_main(int argc, char *argv[]);
    int thrust_main(int argc, char *argv[]);
    int cuda_main(int argc, char *argv[]);

    int retval = EXIT_SUCCESS;
    try
    {
		std::cout << "Boost.Compute library -----------------------------------------------" << std::endl;
		//	REF [library] >> Boost library.

		std::cout << "\nVexCL library -------------------------------------------------------" << std::endl;
        //retval = vexcl_main(argc, argv);  // Not yet implemented.

		std::cout << "\nThrust library ------------------------------------------------------" << std::endl;
        //retval = thrust_main(argc, argv);

		std::cout << "\nCompute Unified Device Architecture (CUDA) --------------------------" << std::endl;
        retval = cuda_main(argc, argv);
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
