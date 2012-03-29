#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char* argv[])
{
	void vector_arithmetic();
	void matrix_arithmetic();
	void cube_arithmetic();

	try
	{
		//vector_arithmetic();
		//matrix_arithmetic();
		cube_arithmetic();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::flush;
	std::cin.get();

    return 0;
}

