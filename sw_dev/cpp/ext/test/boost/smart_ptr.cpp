#include <boost/smart_ptr.hpp>
#include <list>
#include <iostream>
#include <memory>


class Integer
{
public :
	Integer(int i)
	: i_(i)
	{  std::cout << "ctor is called\n";  }
	~Integer()
	{  std::cout << "dtor is called\n";  }

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


void smart_ptr()
{
	//-----------------------------------------------------------------------------------
	//  test 1:
	std::cout << "assign operator & use_count(), unique()" << std::endl;
	{
		boost::shared_ptr<Integer> a(new Integer(100));
		boost::shared_ptr<Integer> b(new Integer(200));

		a = b;
		std::cout << a->get() << '\n';
		std::cout << b->get() << '\n';

		std::cout << b.use_count() << '\n';
		std::cout << b.unique() << '\n';
	}

	std::cout.flush();
	std::cin.get();

	//-----------------------------------------------------------------------------------
	//  test 2:
	std::cout << "with STL container" << std::endl;
	{
		std::list<boost::shared_ptr<Integer> > ctr;

		{
			boost::shared_ptr<Integer> a(new Integer(100));
			ctr.push_back(a);
			std::cout << a.use_count() << '\n';
		}
		std::cout << ctr.front().use_count() << '\n';

		std::cout << "-------------------\n";

		ctr.push_back(boost::shared_ptr<Integer>(new Integer(200)));
		ctr.pop_back();

		std::cout << "-------------------\n";

		ctr.push_back(boost::shared_ptr<Integer>(new Integer(300)));
		ctr.push_back(boost::shared_ptr<Integer>(new Integer(400)));
		ctr.clear();

		std::cout << "-------------------\n";

		ctr.push_back(boost::shared_ptr<Integer>(new Integer(500)));
	}

	std::cout.flush();
	std::cin.get();

	//-----------------------------------------------------------------------------------
	//  test 3: assignment operator & equality operator
	std::cout << "assignment operator & equality operator" << std::endl;
	{
		boost::shared_ptr<Public> a(new Public(1, 2L, 3.0f, 4.0f));
		boost::shared_ptr<Public> b(a);

		std::cout << a->i << "  :  " << a->l << "  :  " << a->f << "  :  " << a->d << '\n';
		std::cout << b->i << "  :  " << b->l << "  :  " << b->f << "  :  " << b->d << '\n';

		std::cout << (a == b) << '\n';

		//
		b->i = -1;

		std::cout << a->i << "  :  " << a->l << "  :  " << a->f << "  :  " << a->d << '\n';
		std::cout << b->i << "  :  " << b->l << "  :  " << b->f << "  :  " << b->d << '\n';

		std::cout << (a == b) << '\n';
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
		if (sp1.get()) std::cout << *sp1 << '\n';
		else std::cout << "null\n";

		if (ap2.get()) std::cout << *ap2 << "  :  ";
		else std::cout << "null  :  ";
		if (sp2.get()) std::cout << *sp2 << '\n';
		else std::cout << "null\n";
	}
}
