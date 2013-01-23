#include <boost/smart_ptr.hpp>
#include <list>
#include <iostream>
#include <memory>


namespace {
namespace local {

class Integer
{
public :
	Integer(int i)
	: i_(i)
	{  std::cout << "ctor is called" << std::endl;  }
	~Integer()
	{  std::cout << "dtor is called" << std::endl;  }

	int get()  {  return i_;  }

private :
	int i_;
};

class Public
{
public:
	Public(int i_ = 0, long l_ = 0L, float f_ = 0.0f, double d_ = 0.0)
	: i(i_), l(l_), f(f_), d(d_)
	{}

public :
	int i;
	long l;
	float f;
	double d;
};

}  // namespace local
}  // unnamed namespace

void smart_ptr()
{
	//-----------------------------------------------------------------------------------
	//  test 1:
	std::cout << "assign operator & use_count(), unique()" << std::endl;
	{
		boost::shared_ptr<local::Integer> a(new local::Integer(100));
		boost::shared_ptr<local::Integer> b(new local::Integer(200));

		a = b;
		std::cout << a->get() << std::endl;
		std::cout << b->get() << std::endl;

		std::cout << b.use_count() << std::endl;
		std::cout << b.unique() << std::endl;
	}

	std::cout.flush();
	std::cin.get();

	//-----------------------------------------------------------------------------------
	//  test 2:
	std::cout << "with STL container" << std::endl;
	{
		std::list<boost::shared_ptr<local::Integer> > ctr;

		{
			boost::shared_ptr<local::Integer> a(new local::Integer(100));
			ctr.push_back(a);
			std::cout << a.use_count() << std::endl;
		}
		std::cout << ctr.front().use_count() << std::endl;

		std::cout << "-------------------" << std::endl;

		ctr.push_back(boost::shared_ptr<local::Integer>(new local::Integer(200)));
		ctr.pop_back();

		std::cout << "-------------------" << std::endl;

		ctr.push_back(boost::shared_ptr<local::Integer>(new local::Integer(300)));
		ctr.push_back(boost::shared_ptr<local::Integer>(new local::Integer(400)));
		ctr.clear();

		std::cout << "-------------------" << std::endl;

		ctr.push_back(boost::shared_ptr<local::Integer>(new local::Integer(500)));
	}

	std::cout.flush();
	std::cin.get();

	//-----------------------------------------------------------------------------------
	//  test 3: assignment operator & equality operator
	std::cout << "assignment operator & equality operator" << std::endl;
	{
		boost::shared_ptr<local::Public> a(new local::Public(1, 2L, 3.0f, 4.0f));
		boost::shared_ptr<local::Public> b(a);

		std::cout << a->i << "  :  " << a->l << "  :  " << a->f << "  :  " << a->d << std::endl;
		std::cout << b->i << "  :  " << b->l << "  :  " << b->f << "  :  " << b->d << std::endl;

		std::cout << (a == b) << std::endl;

		//
		b->i = -1;

		std::cout << a->i << "  :  " << a->l << "  :  " << a->f << "  :  " << a->d << std::endl;
		std::cout << b->i << "  :  " << b->l << "  :  " << b->f << "  :  " << b->d << std::endl;

		std::cout << (a == b) << std::endl;
	}

	//-----------------------------------------------------------------------------------
	//  test 4: with auto_ptr
	std::cout << "with auto_ptr" << std::endl;
	{
		std::auto_ptr<int> ap1(new int (100)), ap2(new int (200));
		boost::shared_ptr<int> sp1(ap1), sp2;
		sp2 = ap2;

		if (ap1.get()) std::cout << *ap1 << "  :  ";
		else std::cout << "null  :  ";
		if (sp1.get()) std::cout << *sp1 << std::endl;
		else std::cout << "null" << std::endl;

		if (ap2.get()) std::cout << *ap2 << "  :  ";
		else std::cout << "null  :  ";
		if (sp2.get()) std::cout << *sp2 << std::endl;
		else std::cout << "null" << std::endl;
	}
}
