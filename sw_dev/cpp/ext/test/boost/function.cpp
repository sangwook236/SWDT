#include <boost/function.hpp>
#include <boost/function_equal.hpp>
#include <boost/bind.hpp>
#include <iostream>


namespace {
namespace local {

int add_two_values(int a, int b)
{
	std::cout << a << " + " << b << " = " << a + b << std::endl; 
    return a + b;
}

int add_three_values(int a, int b, int c)
{
	std::cout << a << " + " << b << " + " << c << " = " << a + b + c << std::endl; 
    return a + b + c;
}

int sub_two_values(int a, int b)
{
	std::cout << a << " - " << b << " = " << a - b << std::endl; 
    return a - b;
}

struct int_div
{ 
  float operator()(int x, int y) const
  {  return float(x) / y;  }; 
};

struct int_sum
{
	typedef void result_type;

    void operator()(int x)  {  sum += x;  }

	int sum;
};

struct X
{
	bool func(int a)
	{
		std::cout << "X::func(" << a << ") is called" << std::endl;
		return true;
	}
};

bool compute_with_X(X*, int)
{
	return true;
}

void function_func()
{
	//
	{
		boost::function<int (int, int)> f;  // is equal to boost::function2<float, int, int> f;
		try
		{
			f(1, 1);  // empty function call
		}
		catch (const boost::bad_function_call &e)
		{
			std::cout << e.what() << std::endl;
		}

		f = &add_two_values;  // free function pointer
		if (f)
			f(1, 1);
		else
			std::cout << "f has no target, so it is unsafe to call" << std::endl;

		f.clear();
		if (!f.empty())
			f(2, 2);
		else
			std::cout << "f has no target, so it is unsafe to call" << std::endl;
	}

	{
		boost::function<int (int, int)> f1 = boost::bind(add_two_values, _2, _1);
		boost::function<int (int param)> f2 = boost::bind(add_three_values, _1, 9, _1);

		const int x = 3;
		const int y = 2;
		const int z = -4;

		f1(x, y);  // y + x
		f2(z);  // z + 9 + z
	}
}

void function_func_obj()
{
	//
	{
		boost::function<float (int x, int y)> f;
		f = int_div();

		std::cout << f(5, 3) << std::endl;
	}

	//
	{
		// In some cases it is expensive (or semantically incorrect) to have Boost.Function clone a function object.
		// In such cases, it is possible to request that Boost.Function keep only a reference to the actual function object.
		// This is done using the ref and cref functions to wrap a reference to a function object.
		int_sum fo;

		// g1 will not make a copy of fo, nor will g2 when it is targeted to g1's reference to fo
		boost::function<void (int)> g1 = boost::ref(fo);
		boost::function<void (int)> g2(fo);

		fo.sum = 0;
		const int a[] = { 1, 2, 3 };
		std::for_each(a, a + 3, boost::bind(boost::ref(fo), _1));

		std::cout << std::boolalpha << (fo.sum == 6) << std::endl;
	}
}

void function_mem_func()
{
	X x;
	X *p = &x;

	// member function call
	{
		boost::function<bool (int)> f1 = std::bind1st(std::mem_fun(&X::func), &x);
		f1(5);  // call x.func(5)

		boost::function<bool (X*, int)> f2 = &X::func;
		f2(&x, -5);
	}

	//
	{
		boost::function<bool (int)> f1 = boost::bind(&X::func, x, _1);					// (internal copy of x).func(...)
		boost::function<bool (int)> f2 = boost::bind(&X::func, p, _1);					// (internal copy of p)->func(...)
		boost::function<bool (int)> f3 = boost::bind(&X::func, boost::ref(x), _1);		// x.func(...)
		boost::function<bool (int)> f4 = boost::bind(&X::func, &x, _1);					// (&x)->func(...)
	}
}

void function_compare()
{
	//
	{
		boost::function<bool (X*, int)> f = &X::func;
		std::cout << std::boolalpha << (f == &X::func) << std::endl;
		std::cout << std::boolalpha << (&compute_with_X == f) << std::endl;
	}

	//
	{
		int_sum fo1, fo2;
		boost::function<void (int)> f = boost::ref(fo1);
		std::cout << std::boolalpha << (f == boost::ref(fo1)) << std::endl;
		//std::cout << std::boolalpha << (f == fo1) << std::endl;  // only if int_sum is EqualityComparable. EqualityComparable types must have == and != operators.
		std::cout << std::boolalpha << (f == boost::ref(fo2)) << std::endl;
	}

	//
	{
		boost::function<int (int, int)> func11 = &add_two_values;
		boost::function<int (int, int)> func12 = &add_two_values;
		boost::function<int (int, int, int)> func2 = &add_three_values;
		boost::function<int (int, int)> func3 = &sub_two_values;

		std::cout << std::endl;
		std::cout << std::boolalpha << boost::function_equal(func11, &add_two_values) << std::endl;

		//std::cout << std::boolalpha << (func11 == func11) << std::endl;  // compile-time error
		//std::cout << std::boolalpha << boost::function_equal(func11, func11) << std::endl;  // compile-time error
		//std::cout << std::boolalpha << (func11 == func12) << std::endl;  // compile-time error
		//std::cout << std::boolalpha << boost::function_equal(func11, func12) << std::endl;  // compile-time error
		//std::cout << std::boolalpha << boost::function_equal(func11, func2) << std::endl;  // compile-time error
		//std::cout << std::boolalpha << boost::function_equal(func11, func3) << std::endl;  // compile-time error

		std::cout << std::boolalpha << (&func11 == &func11) << std::endl;
		std::cout << std::boolalpha << boost::function_equal(&func11, &func11) << std::endl;
		std::cout << std::boolalpha << (&func11 == &func12) << std::endl;  // Oops !! unexpected result: false
		std::cout << std::boolalpha << boost::function_equal(&func11, &func12) << std::endl;  // Oops !! unexpected result: false
		std::cout << std::boolalpha << boost::function_equal(func11, &func12) << std::endl;  // Oops !! unexpected result: false
		std::cout << std::boolalpha << boost::function_equal(&func11, func12) << std::endl;  // Oops !! unexpected result: false
		//std::cout << std::boolalpha << boost::function_equal(&func11, &func2) << std::endl;  // compile-time error
		std::cout << std::boolalpha << boost::function_equal(&func11, &func3) << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void function()
{
	local::function_func();
	local::function_func_obj();
	local::function_mem_func();
	local::function_compare();
}
