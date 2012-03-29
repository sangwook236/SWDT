#include <boost/cast.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <iostream>
#include <string>

namespace
{
    struct Base
    {
		virtual ~Base()  {}
        virtual char kind()  { return 'B'; }
    };

    struct Base2
    {
		virtual ~Base2()  {}
        virtual char kind2()  { return '2'; }
    };

    struct Derived : public Base, Base2
    {
        virtual char kind()  { return 'D'; }
    };
}  // namespace

void conversion()
{
	// polymorphic cast
	{
		Base* base = new Derived;
		Base2* base2 = 0;
		Derived* derived = 0;
		derived = boost::polymorphic_downcast<Derived *>(base);  // downcast
		assert(derived->kind() == 'D');

		derived = 0;
		derived = boost::polymorphic_cast<Derived *>(base); // downcast, throw on error
		assert(derived->kind() == 'D');

		base2 = boost::polymorphic_cast<Base2 *>(base); // crosscast
		assert(base2->kind2() == '2');
		delete base2;

		//  tests which should result in errors being detected
		int err_count = 0;
		base = new Base;

		//derived = boost::polymorphic_downcast<Derived *>(base);  // #1 assert failure

		try
		{
			derived = boost::polymorphic_cast<Derived *>(base);
		}
		catch (const std::bad_cast&)
		{
			std::cout << "caught bad_cast\n";
		}

		delete base;
	}

	// lexical cast
	{
		try
		{
			// string -> number
			const int i = boost::lexical_cast<int>("16");
			std::cout << i << std::endl;

			// number -> string
			const std::string s = boost::lexical_cast<std::string>(321.5f);
			std::cout << s << std::endl;
		}
		catch (const boost::bad_lexical_cast &)
		{
		}
	}

	// numeric cast ==> improved (using boost::numeric::converter)
	{
		try
		{
			const int i = 42;
			const short s = boost::numeric_cast<short>(i);  // This conversion succeeds (is in range)
		}
		catch (const boost::numeric::negative_overflow& e)
		{
			std::cout << "exception #1: " << e.what() << std::endl;
		}
		catch (const boost::numeric::positive_overflow& e)
		{
			std::cout << "exception #2: " << e.what() << std::endl;
		}

		const float f = -42.1234f;
		try
		{
			// This will cause a boost::numeric::negative_overflow exception to be thrown
			const unsigned int i = boost::numeric_cast<unsigned int>(f);
		}
		catch (const boost::numeric::bad_numeric_cast& e)
		{
			std::cout << "exception #3: " << e.what() << std::endl;
		}

		const double d = f + boost::numeric_cast<double>(123);  // int -> double

		const unsigned long l = std::numeric_limits<unsigned long>::max();

		try
		{
			// This will cause a boost::numeric::positive_overflow exception to be thrown
			// NOTE: *operations* on unsigned integral types cannot cause overflow
			// but *conversions* to a signed type ARE range checked by numeric_cast.

			const unsigned char c = boost::numeric_cast<unsigned char>(l);
		}
		catch (const boost::numeric::positive_overflow& e)
		{
			std::cout << "exception #4: " << e.what() << std::endl;
		}
	}

	// boost numeric conversion library
	{
		const int x = boost::numeric::converter<int, double>::convert(2.0);
		assert(x == 2);

		const int y = boost::numeric::converter<int, double>()(3.14);  // As a function object.
		assert(y == 3);  // The default rounding is trunc.

		try
		{
			const double m = boost::numeric::bounds<double>::highest();
			const int z = boost::numeric::converter<int, double>::convert(m);  // By default throws positive_overflow()
		}
		catch (boost::numeric::positive_overflow const & e)
		{
			std::cout << "exception #1: " << e.what() << std::endl;
		}
	}
}
