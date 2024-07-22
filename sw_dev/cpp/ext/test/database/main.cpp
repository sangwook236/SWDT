//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int sqlite_main(int argc, char *argv[]);
	int mysql_main(int argc, char *argv[]);
	int lmdbxx_main(int argc, char *argv[]);

	int soci_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "SQLite --------------------------------------------------------------" << std::endl;
		//retval = sqlite_main(argc, argv);

		std::cout << "\nMySQL ---------------------------------------------------------------" << std::endl;
		//retval = mysql_main(argc, argv);

		std::cout << "\nLMDB++ --------------------------------------------------------------" << std::endl;
		//retval = lmdbxx_main(argc, argv);

		std::cout << "\nSOCI library --------------------------------------------------------" << std::endl;
		retval = soci_main(argc, argv);
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
