//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char *argv[])
{
	int fixed_size(int, char *[]);
	int dynamic_size(int, char *[]);
	int fixed_block(int, char **);
	int dynamic_block(int, char **);
	int coefficient_wise_unary_operator(int, char **);
	int coefficient_wise_biary_operator(int, char **);

	try
	{
		void lu();
		void evd();
		void svd();
		void qr();
		void cholesky();

		//fixed_block(argc, argv);
		//example_dynamic_size(argc, argv);
		//example_dynamic_size(argc, argv);
		//dynamic_block(argc, argv);
		//coefficient_wise_unary_operator(argc, argv);
		//coefficient_wise_biary_operator(argc, argv);

		//lu();
		//evd();
		//svd();
		//qr();
		cholesky();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred: " << std::endl;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return 0;
}
