#include <boost/python.hpp>


/*
in python interperter:
	import greeting_using_boost
	print greeting_using_boost.greet()
*/

char const * greet()
{
   return "hello, world";
}

// module setting
BOOST_PYTHON_MODULE(greeting_using_boost)  // this name has to agree with the module name in setup.py
{
	// method setting
    boost::python::def("greet", greet);
}
