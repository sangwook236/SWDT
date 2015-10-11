#include <boost/functional/value_factory.hpp>
#include <boost/functional/factory.hpp>
#include <boost/function.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/bind.hpp>
#include <map>
#include <iostream>


namespace {
namespace local {

struct Base
{
	virtual void foo() const = 0;
};

class DerivedA: public Base
{
public:
	DerivedA()
	{
		std::cout << "DerivedA() is called" << std::endl;
	}
	explicit DerivedA(const int i)
	{
		std::cout << "DerivedA(" << i << ") is called" << std::endl;
	}
	DerivedA(const int i, const double d)
	{
		std::cout << "DerivedA(" << i << "," << d << ") is called" << std::endl;
	}

	/*virtual*/ void foo() const
	{
		std::cout << "DerivedA::foo() is called" << std::endl;
	}
};

struct DerivedB: public Base
{
	DerivedB()
	{
		std::cout << "DerivedB() is called" << std::endl;
	}
	explicit DerivedB(const int i)
	{
		std::cout << "DerivedB(" << i << ") is called" << std::endl;
	}
	DerivedB(const int i, const long l, const double d)
	{
		std::cout << "DerivedB(" << i << "," << l << "," << d << ") is called" << std::endl;
	}

	/*virtual*/ void foo() const
	{
		std::cout << "DerivedB::foo() is called" << std::endl;
	}
};

void value_factory_basic()
{
	//
	boost::function<DerivedA ()> A_factory1 = boost::value_factory<DerivedA>();
	DerivedA aA1(A_factory1());
	//DerivedA aA2(boost::value_factory<DerivedA>()());
	//DerivedA aA3(A_factory1(1, 2.0));  // compile-time error

	boost::function<DerivedA (const int &)> A_factory2 = boost::value_factory<DerivedA>();
	DerivedA aA4(A_factory2(1));
	//DerivedA aA5(boost::value_factory<DerivedA>()(2));
}

void factory_basic()
{
	//
	boost::function<DerivedA * ()> A_factory1 = boost::factory<DerivedA *>();
	boost::scoped_ptr<DerivedA> aA1(A_factory1());
	//boost::scoped_ptr<DerivedA> aA2(boost::factory<DerivedA *>()());  // compile-time error
	//boost::scoped_ptr<DerivedA> aA3(A_factory1(1, 2.0));  // compile-time error

	boost::function<DerivedA * (const int &)> A_factory2 = boost::factory<DerivedA *>();
	boost::scoped_ptr<DerivedA> aA4(A_factory2(2));
	//boost::scoped_ptr<DerivedA> aA5(boost::factory<DerivedA *>()(2));  // compile-time error
}

template<class ValueFactory> 
void do_something_by_value(ValueFactory creator = ValueFactory())
{
	const int a = 1;
	const double b = 2.0;
    typename ValueFactory::result_type x = creator(a, b);

	x.foo();
}

template<class Factory>
void do_something_by_ptr(Factory creator = Factory())
{
	const int a = -1;
	const double b = -2.0;
	typename Factory::result_type ptr = creator(a, b);

	ptr->foo();
}

void factory_as_argument()
{
	do_something_by_value(boost::value_factory<DerivedA>());
	do_something_by_value(boost::bind(boost::value_factory<DerivedB>(), _1, 5, _2));

	do_something_by_ptr(boost::factory<DerivedA *>());
	do_something_by_ptr(boost::bind(boost::factory<DerivedB *>(), _1, -5, _2));
}

void factory_mapper()
{
	typedef boost::function<Base * (const int &)> factory_type;

	std::map<std::string, factory_type> factories;

    factories["DerivedA"] = boost::factory<DerivedA *>();
    factories["DerivedB"] = boost::factory<DerivedB *>();

	factories["DerivedA"](1)->foo();
	factories["DerivedB"](2)->foo();
}

}  // namespace local
}  // unnamed namespace

void factory()
{
	std::cout << "--------------------------------------------" << std::endl;
	local::value_factory_basic();
	local::factory_basic();

	std::cout << "\n--------------------------------------------" << std::endl;
	local::factory_as_argument();

	std::cout << "\n--------------------------------------------" << std::endl;
	local::factory_mapper();
}
