#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char *argv[])
{
	void matrix_operation();
	void matrix_function();
	void vector_operation();
	void vector_function();
	void lu();
	void cholesky();
	void qr();
	void eigen();
	void svd();

	try
	{
		//matrix_operation();
		//matrix_function();
		//vector_operation();
		vector_function();

		//lu();
		//cholesky();
		//qr();
		//eigen();
		//svd();
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
