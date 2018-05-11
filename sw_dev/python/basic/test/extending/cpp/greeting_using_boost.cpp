#include <boost/python.hpp>


// Usage in Python:
//	import greeting_using_boost
//	print greeting_using_boost.greet()

char const * greet()
{
   return "hello, world";
}

// Module setting.
BOOST_PYTHON_MODULE(greeting_using_boost)  // This name has to agree with the module name in setup.py.
{
	// Method setting.
    boost::python::def("greet", greet);
}
