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
	int newmat_main(int argc, char* argv[]);
	int cvm_main(int argc, char* argv[]);
	int mtl_main(int argc, char* argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "Linear Algebra PACKage (LAPACK) -------------------------------------" << std::endl;
		//retval = lapack_main(argc, argv);

		std::cout << "\nAutomatically Tuned Linear Algebra Software (ATLAS) -----------------" << std::endl;
		retval = atlas_main(argc, argv);
		std::cout << "\nEigen library -------------------------------------------------------" << std::endl;
		//retval = eigen_main(argc, argv);
		std::cout << "\nArmadillo library ---------------------------------------------------" << std::endl;
		//retval = armadillo_main(argc, argv);

		std::cout << "\nNewmat C++ matrix library -------------------------------------------" << std::endl;
		//retval = newmat_main(argc, argv);  // not yet implemented.
		std::cout << "\nCVM Class Library ---------------------------------------------------" << std::endl;
		//retval = cvm_main(argc, argv);

		std::cout << "\nMatrix Template Library (MTL) ---------------------------------------" << std::endl;
		//retval = mtl_main(argc, argv);
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
