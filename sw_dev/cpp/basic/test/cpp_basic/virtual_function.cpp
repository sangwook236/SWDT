#include <iostream>


namespace {
namespace local {

class Base
{
public:
	void f1()
	{
		std::cout << "Base::f1() is called" << std::endl;
	}
	void f2()
	{
		std::cout << "Base::f2() is called" << std::endl;
	}
	virtual void vf1()
	{
		std::cout << "Base::vf1() is called" << std::endl;
	}
	virtual void vf2()
	{
		std::cout << "Base::vf2() is called" << std::endl;
	}
};

class Derived: public Base
{
public:
	void f2()
	{
		std::cout << "Derived::f2() is called" << std::endl;
	}
	void f3()
	{
		std::cout << "Derived::f3() is called" << std::endl;
	}
	virtual void vf2()
	{
		std::cout << "Derived::vf2() is called" << std::endl;
	}
	virtual void vf3()
	{
		std::cout << "Derived::vf3() is called" << std::endl;
	}
};
	
}  // namespace local
}  // unnamed namespace

void virtual_function()
{
	local::Base *b1 = new local::Base();
	local::Base *b2 = new local::Derived();
	local::Derived *d = new local::Derived();

	b1->f1();
	b1->f2();
	//b1->f3();  // compile-time error
	b1->vf1();
	b1->vf2();
	//b1->vf3();  // compile-time error

	std::cout << std::endl;

	b2->f1();
	b2->f2();
	//b2->f3();  // compile-time error
	b2->vf1();
	b2->vf2();
	//b2->vf3();  // compile-time error

	std::cout << std::endl;

	d->f1();
	d->f2();
	d->f3();
	d->vf1();
	d->vf2();
	d->vf3();

	delete b1;
	delete b2;
	delete d;
}
