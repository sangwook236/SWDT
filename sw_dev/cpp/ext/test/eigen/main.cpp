//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char *argv[])
{
	void basic_operation();
	void linear_system();

    void lu();
    void evd();
    void svd();
    void qr();
    void cholesky();

	try
	{
		//basic_operation();
		//linear_system();

		//lu();
		//evd();
		//svd();
		qr();
		//cholesky();
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
