#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <sstream>
#include <fstream>
#include <iostream>
#include <set>
#include <typeinfo>
#include <cmath>


namespace {
namespace local {

struct Test
{
	int i;
	long l;
	float f;
	double d;
};

void cpp_2011()
{
	const unsigned long long ull = 18'446'744'073'709'550'592ull;
	std::cout << ull << std::endl;

	std::cout << std::hexfloat << 0.01 << std::endl;
}

void cpp_2014()
{
}

}  // namespace local
}  // unnamed namespace


int main(int argc, char **argv)
{
	void virtual_function();
	void predefined_macro();
	void array();
	void string();
	void unicode_string();
	void string_tokenization();
	void complex();
	void date_time();

	void file_io();

	void stl_data_structure();
	void stl_algorithm();

	void performance_analysis();

	void static_lib_linking();
	void shared_lib_linking();

	void deprecated();

	try
	{
		//local::cpp_2011();
		//local::cpp_2014();

		//virtual_function();

		//predefined_macro();
		//array();
		//string();
		unicode_string();
		//string_tokenization();
		//complex();
		//date_time();

		//file_io();  // Not yet implemented.

		//stl_data_structure();
		//stl_algorithm();

		//performance_analysis();

		//
		//static_lib_linking();
		//shared_lib_linking();

		//deprecated();

		// Test.
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
	catch (const std::exception &ex)
	{
		std::cerr << "std::exception caught !!!: " << ex.what() << std::endl;
	}
	catch (...)
	{
		std::cerr << "Unknown exception caught !!!" << std::endl;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

    return 0;
}
