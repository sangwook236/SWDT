#include "../Complex.h"
#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>


class ComplexTest: public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(ComplexTest);
	CPPUNIT_TEST(testEquality);
	CPPUNIT_TEST(testAddition);
	CPPUNIT_TEST_EXCEPTION(testDivideByZeroThrows, std::runtime_error);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()
	{
		m_10_1 = new Complex<double>(10, 1);
		m_1_1 = new Complex<double>(1, 1);
		m_11_2 = new Complex<double>(11, 2);
	}

	void tearDown()
	{
		delete m_10_1;
		delete m_1_1;
		delete m_11_2;
	}

	void testEquality()
	{
		CPPUNIT_ASSERT(*m_10_1 == *m_10_1);
		CPPUNIT_ASSERT(!(*m_10_1 == *m_11_2));
	}

	void testAddition()
	{
		CPPUNIT_ASSERT(*m_10_1 + *m_1_1 == *m_11_2);
	}

	void testDivideByZeroThrows()
	{
		// The following line should throw a MathException.
		try
		{
			Complex<double> c = Complex<double>();
			Complex<double> cc = *m_10_1 / Complex<double>(0);
		}
		catch (...)
		{
			throw std::runtime_error("divide by zero");
		}
	}

private:
	Complex<double> *m_10_1, *m_1_1, *m_11_2;
};

CPPUNIT_TEST_SUITE_REGISTRATION(ComplexTest);
//CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(ComplexTest, "Complex Test");  // not working
