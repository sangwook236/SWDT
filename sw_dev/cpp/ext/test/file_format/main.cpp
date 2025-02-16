//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int string_tokenization_main(int argc, char *argv[]);
	int csv_parser_main(int argc, char *argv[]);

	int assimp_main(int argc, char *argv[]);
	int hdf_main(int argc, char *argv[]);
	int arff_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "string tokenizer ----------------------------------------------------" << std::endl;
		//retval = string_tokenization_main(argc, argv);

		std::cout << "\nCSV Parser library --------------------------------------------------" << std::endl;
		//retval = csv_parser_main(argc, argv);

		std::cout << "\nASSIMP library ------------------------------------------------------" << std::endl;
		//retval = assimp_main(argc, argv);  // Not yet implemented.

		std::cout << "\nHDF library ---------------------------------------------------------" << std::endl;
		//retval = hdf_main(argc, argv);  // Not yet implemented.

		std::cout << "\nARFF file reader ----------------------------------------------------" << std::endl;
		retval = arff_main(argc, argv);
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
