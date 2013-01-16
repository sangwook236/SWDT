#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char* argv[])
{
	int lapack_main(int argc, char* argv[]);
	int atlas_main(int argc, char* argv[]);
	int eigen_main(int argc, char* argv[]);
	int armadillo_main(int argc, char* argv[]);
	int cvm_main(int argc, char* argv[]);
	int mtl_main(int argc, char* argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		lapack_main(argc, argv);
		
		atlas_main(argc, argv);
		eigen_main(argc, argv);
		armadillo_main(argc, argv);
		
		cvm_main(argc, argv);

		//mtl_main(argc, argv);  // not yet implemented
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc occurred: " << e.what() << std::endl;
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
