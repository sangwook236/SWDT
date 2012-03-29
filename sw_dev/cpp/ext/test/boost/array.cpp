#include <boost/array.hpp>
#include <functional>
#include <string>
#include <iostream>


namespace {

template <class T>
inline void print_elements(const T &coll, const char *optcstr = "")
{
    std::cout << optcstr;
    for (typename T::const_iterator it = coll.begin(); it != coll.end(); ++it)
        std::cout << *it << ' ';
    std::cout << std::endl;
}

}  // unnamed namespace

void array_()
{
	//-----------------------------------------------------------------------------------
	//  test 1: basic operations 1
	std::cout << "basic operations 1" << std::endl;
	{
		// define special type name
		typedef boost::array<float, 6> array_type;

		// create and initialize an array
		array_type a = { { 42 } };

		// access elements
		for (unsigned i = 1; i < a.size(); ++i)
			a[i] = a[i-1] + 1;

		// use some common STL container operations
		std::cout << "size:     " << a.size() << std::endl;
		std::cout << "empty:    " << (a.empty() ? "true" : "false") << std::endl;
		std::cout << "max_size: " << a.max_size() << std::endl;
		std::cout << "front:    " << a.front() << std::endl;
		std::cout << "back:     " << a.back() << std::endl;
		std::cout << "elems:    ";
		// iterate through all elements
		for (array_type::const_iterator it = a.begin(); it < a.end(); ++it)
			std::cout << *it << ' ';
		std::cout << std::endl;

		// check copy constructor and assignment operator
		array_type b(a);
		array_type c;
		c = a;
		if (a==b && a==c)
			std::cout << "copy construction and copy assignment are OK" << std::endl;
		else
			std::cout << "copy construction and copy assignment FAILED" << std::endl;
	}

	//-----------------------------------------------------------------------------------
	//  test 2: basic operations 2
	std::cout << "basic operations 2" << std::endl;
	{
		typedef boost::array<std::string, 4> array_type;

		// create array of four seasons
		array_type seasons = { { "spring", "summer", "autumn", "winter" } };

		// copy and change order
		array_type seasons_orig = seasons;
		for (unsigned i = seasons.size() - 1; i > 0; --i)
			std::swap(seasons.at(i), seasons.at((i+1) % seasons.size()));

		std::cout << "one way:   ";
		print_elements(seasons);

		// try swap()
		std::cout << "other way: ";
		std::swap(seasons, seasons_orig);
		print_elements(seasons);

		// try reverse iterators
		std::cout << "reverse:  ";
		for (array_type::reverse_iterator rit = seasons.rbegin(); rit < seasons.rend(); ++rit)
			std::cout << " " << *rit;
		std::cout << std::endl;
	}

	//-----------------------------------------------------------------------------------
	//  test 3: algorithms
	std::cout << "algorithms" << std::endl;
	{
		// create and initialize array
		boost::array<int, 10> a = { { 1, 2, 3, 4, 5 } };
		print_elements(a);

		// modify elements directly
		for (unsigned i = 0; i < a.size(); ++i)
			++a[i];
		print_elements(a);

		// change order using an STL algorithm
		std::reverse(a.begin(), a.end());
		print_elements(a);

		// negate elements using STL framework
		std::transform(a.begin(), a.end(),  // source
			a.begin(),                      // destination
			std::negate<int>()              // operation
		);
		print_elements(a);
	}
}
