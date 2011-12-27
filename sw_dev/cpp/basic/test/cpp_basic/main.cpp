#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <typeinfo>


struct Test
{
	int i;
	long l;
	float f;
	double d;
};

int main(int argc, char *argv[])
{
	void test_predefined_macro();
	void test_array();
	void test_complex();
	void test_date_time();
	void stl_algorithm();
	void virtual_function();

	//test_predefined_macro();
	//test_array();
	//test_complex();
	//test_date_time();
	stl_algorithm();
	//virtual_function();

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

	std::wcout << L"press any key to exit ..." << std::endl;
	std::wcout.flush();
	std::wcin.get();

	return 0;
}
