#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


void bind_func();
void bind_func_obj();
void bind_mem_func_1();
void bind_mem_func_2();

void bind()
{
	bind_func();
	bind_func_obj();
	bind_mem_func_1();
	bind_mem_func_2();
}

namespace
{

int func_1(int a, int b)
{
	std::cout << a << " + " << b << " = " << a + b << std::endl; 
    return a + b;
}

int func_2(int a, int b, int c)
{
	std::cout << a << " + " << b << " + " << c << " = " << a + b + c << std::endl; 
    return a + b + c;
}

struct FuncObj
{
	typedef void result_type;

	int operator()(int a, int b)  {  return a - b;  }
	bool operator()(long a, long b)  {  return a == b;  }

    void operator()(int x)  {  sum += x;  }

	int sum;
};

class player
{
public:
	void play1(const int &i)
	{
		std::cout << "player::play1(" << i << ") is called" << std::endl;
	}
	void play2(const int i)
	{
		std::cout << "player::play2(" << i << ") is called" << std::endl;
	}
	void stop()
	{
		std::cout << "player::stop() is called" << std::endl;
	}

	bool take_rest(int i, long l)
	{
		return true;
	}
};

struct X
{
	bool func(int a)
	{
		std::cout << "X::func(" << a << ") is called" << std::endl;
		return true;
	}
};

}  // unnamed namespace

void bind_func()
{
	const int x = 3;
	const int y = 2;
	const int z = -4;

	//
	boost::bind(func_1, _2, _1)(y, x);			// func_1(y, x)
	boost::bind(func_2, _1, 9, _1)(x);			// func_2(x, 9, x)
	boost::bind(func_2, _3, _3, _3)(x, y, z);	// func_2(z, z, z)
	boost::bind(func_2, _1, _1, _1)(x, y, z);	// func_2(x, x, x)

	//
	boost::function<int (int, int)> gg1 = boost::bind(func_1, _2, _1);
	boost::function<int (int)> gg2 = boost::bind(func_2, _1, 9, _1);

	gg1(x, y);  // y + x
	gg2(z);  // z + 9 + z

	//
	const int i = 5;
	boost::bind(func_1, i, _1);					// a copy of the value of i is stored into the function object
	boost::bind(func_1, boost::ref(i), _1);		// boost::ref and boost::cref can be used to make the function object store a reference to an object
	boost::bind(func_1, boost::cref(42), _1);
}

void bind_func_obj()
{
	// bind makes a copy of the provided function object
	FuncObj fo;
	fo.sum = 0;

	const int x = 104;
	boost::bind<int>(fo, _1, _1)(x);  // fo(x, x), i.e. zero
	boost::bind(boost::type<int>(), fo, _1, _1)(x);  // a compiler have trouble with the bind<return_type>(fo, ...) syntax

	// boost::ref and boost::cref can be used to make it store a reference to the function object, rather than a copy
	const int a[] = { 1, 2, 3 };
	std::for_each(a, a + 3, boost::bind(boost::ref(fo), _1));

	std::cout << std::boolalpha << (fo.sum == 6) << std::endl;
}

void bind_mem_func_1()
{
	player thePlayer;

	//
	boost::bind(&player::take_rest, &thePlayer, _1, _2)(1, 2L);
	//is equivalent to
	boost::bind<bool>(boost::mem_fn(&player::take_rest), &thePlayer, _1, _2)(1, 2L);  // boost::bind<return_type>(boost::mem_fn(&Class::mem_func), args);

	//
	//boost::function<void (int &)> playButton1 = boost::bind(&player::play1, &thePlayer, _1);  // compile-time error
	boost::function<void (int)> playButton1 = boost::bind(&player::play1, &thePlayer, _1);
	boost::function<void (int)> playButton2 = boost::bind(&player::play2, &thePlayer, _1);
	boost::function<void ()> stopButton = boost::bind(&player::stop, &thePlayer);

	playButton1(3);
	playButton2(3);
	stopButton();
}

void bind_mem_func_2()
{
	X x;

	boost::shared_ptr<X> p(new X);

	const int i = 5;

	boost::bind(&X::func, boost::ref(x), _1)(i);		// x.func(i)
	boost::bind(&X::func, &x, _1)(i);					// (&x)->func(i)
	boost::bind(&X::func, x, _1)(i);					// (internal copy of x).func(i)
	boost::bind(&X::func, p, _1)(i);					// (internal copy of p)->func(i)
}
