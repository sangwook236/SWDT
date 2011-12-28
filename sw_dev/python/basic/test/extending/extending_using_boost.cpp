#include "stdafx.h"
#include <boost/python.hpp>


char const * greet()
{
   return "hello, world";
}

BOOST_PYTHON_MODULE(hello_module)
{
    boost::python::def("greet", greet);
}

/*
in python interperter:
	import hello_module
	print hello_module.greet()
*/
