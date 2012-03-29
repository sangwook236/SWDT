#include <boost/type_traits.hpp>
#include <iostream>


struct A
{
};

class B
{
};

template<typename T>
class C
{
};

void type_traits()
{
	//
	std::cout << boost::is_class<int>::value << std::endl;
	std::cout << boost::is_class<unsigned long>::value << std::endl;
	std::cout << boost::is_class<double>::value << std::endl;
	std::cout << boost::is_class<A>::value << std::endl;
	std::cout << boost::is_class<B>::value << std::endl;
	std::cout << boost::is_class<C<int> >::value << std::endl;
}
