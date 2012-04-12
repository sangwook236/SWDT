#include <boost/type_traits.hpp>
#include <iostream>


namespace {
namespace local {

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

}  // namespace local
}  // unnamed namespace

void type_traits()
{
	//
	std::cout << boost::is_class<int>::value << std::endl;
	std::cout << boost::is_class<unsigned long>::value << std::endl;
	std::cout << boost::is_class<double>::value << std::endl;
	std::cout << boost::is_class<local::A>::value << std::endl;
	std::cout << boost::is_class<local::B>::value << std::endl;
	std::cout << boost::is_class<local::C<int> >::value << std::endl;
}
