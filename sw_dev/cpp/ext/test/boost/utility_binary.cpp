#include "stdafx.h"
#include <boost/utility/binary.hpp>
#include <iostream>
#include <typeinfo>


namespace {

void func(const int &val)
{
	std::cout << "void func(" << typeid(val).name() << ") is called: " << val << std::endl;
}

void func(const unsigned long &val)
{
	std::cout << "void func(" << typeid(val).name() << ") is called: " << val << std::endl;
}

}  // unnamed namespace

void utility_binary()
{
	const int value1 = BOOST_BINARY(100 111000 01 1 110);  // int
	const unsigned long value2 = BOOST_BINARY_UL(100 001);  // unsigned long
	const long long value3 = BOOST_BINARY_LL(11 000);  // long long if supported

	std::cout << value1 << std::endl;
	std::cout << value2 << std::endl;
	std::cout << value3 << std::endl;

	//
	std::cout << std::boolalpha << ((BOOST_BINARY(10010) & BOOST_BINARY(11000)) == BOOST_BINARY(10000)) << std::endl;

	//
	func(BOOST_BINARY(1010));
	func(BOOST_BINARY_LU(1010));
}
