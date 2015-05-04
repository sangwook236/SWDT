#include <iostream>
#include <complex>
#include <typeinfo>


#if defined(_MSC_VER) && _MSC_VER >= 1300
//#	define __FUNCTION__ __FUNCTION__
//#	define __FUNCSIG__ __FUNCSIG__
#elif defined(__GNUC__)
//#	define __FUNCTION__ __FUNCTION__
#	define __FUNCSIG__ __PRETTY_FUNCTION__
#else
#	error unsupported os platform !!!
#endif


namespace {
namespace local {

void display_name(const std::string &function)
{
	const std::string::size_type pos = std::string(function).find_first_of(std::string("::"));

	if (pos == std::string::npos)
	{
		// class name
		std::cout << "class name: " << std::string("") << std::endl;
		// method name
		std::cout << "method name: " << function << std::endl;
	}
	else
	{
		// class name
		std::cout << "class name: " << std::string(function).substr(0, pos) << std::endl;
		// method name
		std::cout << "method name: " << std::string(function).substr(pos + 2) << std::endl;
	}
}

int func(const bool b, const int * const i, double &d)
{
	std::cout << "__FUNCTION__: \t" << __FUNCTION__ << std::endl;
	std::cout << "__FUNCSIG__: \t" << __FUNCSIG__ << std::endl;

	display_name(__FUNCTION__);

	return *i;
}

class MacroObj
{
public:
	int mem_func(const bool b, const int * const i, double &d) const
	{
		std::cout << "__FUNCTION__: \t" << __FUNCTION__ << std::endl;
		std::cout << "__FUNCSIG__: \t" << __FUNCSIG__ << std::endl;

		display_name(__FUNCTION__);
		// class name
		//std::cout << std::string(__FUNCTION__).substr(0, std::string(__FUNCTION__).find_first_of(std::string("::"))) << std::endl;
		// method name
		//std::cout << std::string(__FUNCTION__).substr(std::string(__FUNCTION__).find_first_of(std::string("::")) + 2) << std::endl;

		return *i;
	}
};

}  // namespace local
}  // unnamed namespace

void predefined_macro()
{
	// ANSI-Compliant Predefined Macros
	{
		std::cout << "__FILE__'s type: \t" << typeid(__FILE__).name() << std::endl;
		std::cout << "__LINE__'s type: \t" << typeid(__LINE__).name() << std::endl;
		std::cout << "__DATE__'s type: \t" << typeid(__DATE__).name() << std::endl;
		std::cout << "__TIME__'s type: \t" << typeid(__TIME__).name() << std::endl;
		std::cout << "__TIMESTAMP__'s type: \t" << typeid(__TIMESTAMP__).name() << std::endl;
		//std::cout << "__STDC__'s type: \t" << typeid(__STDC__).name() << std::endl;

		std::cout << "__FILE__: \t" << __FILE__ << std::endl;
		std::cout << "__LINE__: \t" << __LINE__ << std::endl;
		std::cout << "__DATE__: \t" << __DATE__ << std::endl;
		std::cout << "__TIME__: \t" << __TIME__ << std::endl;
		std::cout << "__TIMESTAMP__: \t" << __TIMESTAMP__ << std::endl;
		//std::cout << "__STDC__: " << __STDC__ << std::endl;
	}
	std::cout << std::endl;

	// Non ANSI-Compliant Predefined Macros
	{
		std::cout << "__FUNCTION__'s type: \t" << typeid(__FUNCTION__).name() << std::endl;
		std::cout << "__FUNCSIG__'s type: \t" << typeid(__FUNCSIG__).name() << std::endl;

		bool b = true;
		int i = 1;
		double d = 0.0;

		local::func(b, &i, d);

		local::MacroObj obj;
		obj.mem_func(b, &i, d);
	}
}
