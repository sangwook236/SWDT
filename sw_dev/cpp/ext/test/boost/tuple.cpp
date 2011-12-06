#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_io.hpp>
#include <sstream>
#include <iostream>


void tuple_basic();
void tuple_tier();
void tuple_streaming();

void tuple()
{
	tuple_basic();
	tuple_tier();
	tuple_streaming();
}

namespace {

class A {};
class B : public A {};
struct C { C() {} C(const B&) {} };
struct D { operator C() const { return C(); } };

}  // unnamed namespace

void tuple_basic()
{
	{
		// if no initial value for an element is provided, it is default initialized (and hence must be default initializable)
		boost::tuple<int, double> a;
		boost::tuple<int, double> b(1);
		boost::tuple<int, double> c(1, 3.14);

		std::cout << a << std::endl;
		std::cout << b << std::endl;
		std::cout << c << std::endl;
	}

	{
		double d = 5; 
		//boost::tuple<double&> a(d);  // error
		boost::tuple<double&> a(boost::ref(d));
		//boost::tuple<double&> b (d + 3.14);  // error: cannot initialize non-const reference with a temporary
		boost::tuple<const double&> c(d + 3.14);  // ok, but dangerous: the element becomes a dangling reference

		std::cout << a << std::endl;
		//std::cout << b << std::endl;
		std::cout << c << std::endl;
	}

	// array
	{
		char arr[10] = { '0', };

		//boost::tuple<char [10]> a(arr);  // error, arrays can not be copied
		boost::tuple<char [10]> b;
	}

	// make_tuple
	{
		double val1 = 1.0;
		int val2 = -1;
		const double cval1 = val1;

		boost::tuple<const double&, int> a = boost::make_tuple(boost::cref(val1), val2);
		boost::tuple<double&, int> b = boost::make_tuple(boost::ref(val1), val2);
		boost::tuple<double&, const int&> c = boost::make_tuple(boost::ref(val1), boost::cref(val2)); 
		boost::tuple<const double&> d = boost::make_tuple(boost::cref(cval1));
		boost::tuple<const double&> e = boost::make_tuple(boost::ref(cval1));
	}

	// access
	{
		double d = 2.7;
		long a = 2L;

		boost::tuple<int, double&, const long&> t(1, d, a);
		const boost::tuple<int, double&, const long&> ct = t;

		int i = boost::get<0>(t);
		i = t.get<0>();
		int j = boost::get<0>(ct);
		boost::get<0>(t) = 5;
		//boost::get<0>(ct) = 5;  // error, can't assign to const 

		double e = boost::get<1>(t);
		boost::get<1>(t) = 3.14;
		//boost::get<2>(t) = long();  // error, can't assign to const 
		//long aa = boost::get<3>(t);  // error: index out of bounds 

		++boost::get<0>(t);
	}

	// assignment
	{
		boost::tuple<char, B*, B, D> t;
		boost::tuple<int, A*, C, C> a(t);
		a = t;
	}

	// std::make_pair
	{
		//boost::tuple<float, int> a = std::make_pair(1, 'a');  // don't work: i don't know why
		//std::cout << a << std::endl;
	}
}

void tuple_tier()
{
	int i;
	char c;
	double d; 

	// The above tie function creates a tuple of type tuple<int&, char&, double&>.
	// The same result could be achieved with the call make_tuple(ref(i), ref(c), ref(a)).
	boost::tie(i, c, d) = boost::make_tuple(1, 'a', 5.5);
	std::cout << i << " " << c << " " << d << std::endl;

	boost::tie(i, c) = std::make_pair(1, 'a');
	std::cout << i << " " << c << std::endl;
}

void tuple_streaming()
{
	// out streaming
	{
		boost::tuple<float, int, std::string> a(1.0f,  2, std::string("Howdy folks!"));
		std::cout << a << std::endl;

		std::cout << boost::tuples::set_open('[') << boost::tuples::set_close(']') << boost::tuples::set_delimiter(',') << a << std::endl; 
	}

	// in streaming
	{
		std::istringstream stream("(1 2 3) [4:5]");

		boost::tuple<int, int, int> i;
		boost::tuple<int, int> j;

		stream >> i;
		stream >> boost::tuples::set_open('[') >> boost::tuples::set_close(']') >> boost::tuples::set_delimiter(':');
		stream >> j;

		std::cout << i << std::endl;
		std::cout << j << std::endl;
	}
}
