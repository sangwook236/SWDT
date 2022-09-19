//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int rapidjson_main(int argc, char *argv[]);
	int yaml_cpp_main(int argc, char *argv[]);
	int protobuf_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "Boost.Serialization library -----------------------------------------" << std::endl;
		// REF [library] >> Boost library.

		std::cout << "\nRapidjson library ---------------------------------------------------" << std::endl;
		retval = rapidjson_main(argc, argv);

		std::cout << "\nyaml-cpp library ----------------------------------------------------" << std::endl;
		//retval = yaml_cpp_main(argc, argv);

		std::cout << "\nProtocol Buffers library --------------------------------------------" << std::endl;
		//retval = protobuf_main(argc, argv);
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
		std::cerr << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit..." << std::endl;
	std::cin.get();

	return retval;
}
