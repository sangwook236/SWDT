#include <boost/any.hpp>
#include <string>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void any()
{
	//-----------------------------------------------------------------------------------
	//  test 1: modifiers & queries
	std::cout << "modifiers & queries" << std::endl;
	{
		//  construction
		::boost::any any1, any2(100), any3(-9876.54321);

		//  type(), empty()
		std::cout << any1.type().name() << "  :  " << any2.type().name() << "  :  " << any3.type().name() << std::endl;
		std::cout << any1.empty() << "  :  " << any2.empty() << "  :  " << any3.empty() << std::endl;

		//  assignment
		any1 = any2;
		any3 = 3ul;
		std::cout << any1.type().name() << "  :  " << any3.type().name() << std::endl;

		//  swap()
		any1.swap(any3);
		std::cout << any1.type().name() << "  :  " << any3.type().name() << std::endl;
	}

	//-----------------------------------------------------------------------------------
	//  test 2: casting
	std::cout << "casting" << std::endl;
	{
		//  std::string
		::boost::any aAny = std::string("aaa");
		const std::string* pStr = ::boost::any_cast<std::string>(&aAny);
		if (pStr) std::cout << pStr->c_str() << std::endl;
		else std::cout << "invalid casting\n";

		const int* pInt = ::boost::any_cast<int>(&aAny);
		if (pInt) std::cout << *pInt << std::endl;
		else std::cout << "invalid casting\n";

		//  pointer
		double* ptr = new double(500.0);
		aAny = ptr;

		double** pPtr = ::boost::any_cast<double*>(&aAny);
		if (pPtr) std::cout << **pPtr << std::endl;
		else std::cout << "invalid casting\n";

		int** pIntPtr = ::boost::any_cast<int*>(&aAny);
		if (pIntPtr) std::cout << **pIntPtr << std::endl;
		else std::cout << "invalid casting\n";

		delete ptr;

		//  array
		long array[2] = { -300l, 300l };
		//  compile-time error  :  cannot specify explicit initializer for arrays
		//aAny = array;
		aAny = (long*)array;

		long** pArray = ::boost::any_cast<long*>(&aAny);
		if (pArray) std::cout << **pArray << std::endl;
		else std::cout << "invalid casting\n";
	}

	{
		::boost::any aAny(100);
		try {
			const int& rAny = ::boost::any_cast<int>(aAny);
			std::cout << rAny << std::endl;
		}
		catch (::boost::bad_any_cast& e) {
			std::cout << e.what() << std::endl;
		}

		try {
			const unsigned int& rAny = ::boost::any_cast<unsigned int>(aAny);
			std::cout << rAny << std::endl;
		}
		catch (::boost::bad_any_cast& e) {
			std::cout << e.what() << std::endl;
		}
	}
}
