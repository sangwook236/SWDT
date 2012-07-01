#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <typeinfo>


namespace {
namespace local {

struct Test
{
	int i;
	long l;
	float f;
	double d;
};

}  // namespace local
}  // unnamed namespace


int main(int argc, char **argv)
{
	void virtual_function();
	void test_predefined_macro();
	void test_array();
	void test_complex();
	void test_date_time();
	void file_io();
	void stl_algorithm();

	try
	{
		//virtual_function();

		//test_predefined_macro();
		//test_array();
		//test_complex();
		//test_date_time();

		//file_io();  // not yet implemented

		stl_algorithm();

		// test
#if 0
		{
			const float f11 = -3.45f;
			const float f12 = 2.356324e34f;

			std::cout << f11 << ", " << f12 << std::endl;

			const unsigned int i11 = (unsigned int)f11;
			const unsigned int i12 = (unsigned int)f12;

			std::cout << i11 << ", " << i12 << std::endl;

			const unsigned int i21 = *(unsigned int *)&f11;
			const unsigned int i22 = *(unsigned int *)&f12;

			std::cout << i21 << ", " << i22 << std::endl;

			const float f21 = *(float *)&i21;
			const float f22 = *(float *)&i22;

			std::cout << f21 << ", " << f22 << std::endl;
		}
#endif
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred !!!: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred !!!" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

    return 0;
}
