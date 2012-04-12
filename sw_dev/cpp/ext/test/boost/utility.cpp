#include <boost/utility.hpp>
#include <list>
#include <iostream>


namespace {
namespace local {

struct useless_type {};
class nonaddressable
{
	useless_type operator&() const;
};

}  // namespace local
}  // unnamed namespace

void utility()
{
	// boost::checked_delete(), boost::checked_array_delete(), boost::checked_deleter, boost::checked_array_deleter
	{
	}

	// boost::next(), boost::prior()
	{
/*
		template <class T>
		T next(T x) { return ++x; }

		template <class T, class Distance>
		T next(T x, Distance n)
		{
			std::advance(x, n);
			return x;
		}

		template <class T>
		T prior(T x) { return --x; }

		template <class T, class Distance>
		T prior(T x, Distance n)
		{
			std::advance(x, -n);
			return x;
		}
*/
		std::list<int> data;
		data.push_back(1);
		data.push_back(2);
		data.push_back(3);
		data.push_back(4);
		data.push_back(5);

		const std::list<int>::iterator p = data.begin();
		const std::list<int>::iterator next = boost::next(p, 2);
		const std::list<int>::iterator prev = boost::prior(next);
		std::cout << "next: " << *next << std::endl;
		std::cout << "prev: " << *prev << std::endl;
	}

	// boost::noncopyable
	{
	}

	// boost::addressof
	{
		local::nonaddressable x;
		local::nonaddressable *xp1 = boost::addressof(x);
		//local::nonaddressable *xp2 = &x;  // compile-time error
	}

	// boost::result_of
	{
	}
}
