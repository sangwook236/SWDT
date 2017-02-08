#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <cvm/cvm.h>
#else
#include <cvm.h>
#endif
#include <algorithm>
#include <string>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_cvm {

void matrix_function()
{
	// max & min
	std::cout << ">>> max & min" << std::endl;
	try
	{
		double a[] = { 3., 2., -5., -4., 5., -6. };
		const cvm::rmatrix m(a, 2, 3);

		std::cout << m;
		int ii = m.indofmax();
		//std::cout << m[m.indofmax()] << std::endl;
		//std::cout << m[m.indofmin()] << std::endl;
		std::cout << *std::max_element(m.begin(), m.end()) << std::endl;
		std::cout << *std::min_element(m.begin(), m.end()) << std::endl;
	}
	catch (const cvm::cvmexception &e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// index of max & min
	std::cout << ">>> index of max & min" << std::endl;
	try
	{
		double a[] = { 3., 2., -5., -4., 5., -6. };
		const cvm::rmatrix m(a, 2, 3);

		std::cout << m;
		std::cout << m.indofmax() << std::endl;  // caution !!!: maximum absolute value
		std::cout << m.indofmin() << std::endl;  // caution !!!: minimum absolute value
	}
	catch (const cvm::cvmexception &e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// index of row & column of max & min
	std::cout << "\n>>> index of row & column of max & min" << std::endl;
	try
	{
		double a[] = { 3., 2., -5., -4., 5., -6. };
		const cvm::rmatrix m(a, 2, 3);

		std::cout << m;
		std::cout << m.rowofmax() << std::endl;  // caution !!!: maximum absolute value
		std::cout << m.rowofmin() << std::endl;  // caution !!!: minimum absolute value

		std::cout << m.colofmax() << std::endl;  // caution !!!: maximum absolute value
		std::cout << m.colofmin() << std::endl;  // caution !!!: minimum absolute value
	}
	catch (const cvm::cvmexception &e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// norm
	std::cout << "\n>>> norm" << std::endl;
	try
	{
		double a[] = { 1., 2., 3., -4., 5., -6. };
		const cvm::rmatrix m(a, 2, 3);

		std::cout << m;
		std::cout << m.norm() << std::endl;
		std::cout << m.norminf() << std::endl;
		std::cout << m.norm1() << std::endl;
		std::cout << m.norm2() << std::endl;
	}
	catch (const cvm::cvmexception &e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// det
	std::cout << "\n>>> det" << std::endl;
	try
	{
		double a[] = { 1., 2., 3., 4., 5., 6., 7., 8., 10. };

		cvm::srmatrix m(a, 3);
		std::cout << m << std::endl << m.det() << std::endl;
	}
	catch (const cvm::cvmexception &e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// inverse
	std::cout << "\n>>> inverse" << std::endl;
	try
	{
		// non-square matrix: pseudo-inverse
		cvm::rmatrix mA(3, 4);
		mA(1,1) = 1.; mA(1,2) = -1.; mA(1,3) = 2.; mA(1,4) = 0.;
		mA(2,1) = -1.; mA(2,2) = 2.; mA(2,3) = -3.; mA(2,4) = 1.;
		mA(3,1) = 0.; mA(3,2) = 1.; mA(3,3) = -1.; mA(3,4) = 1.;

		cvm::rmatrix mX = mA.pinv(1.e-13);
		std::cout << mX << (mA * mX * mA - mA).norm2() << std::endl;

		// square matrix: inverse
		cvm::srmatrix mB(3);
		mB(1,1) = 1.; mB(1,2) = -1.; mB(1,3) = 2.;
		mB(2,1) = -1.; mB(2,2) = 2.; mB(2,3) = -3.;
		mB(3,1) = 0.; mB(3,2) = 1.; mB(3,3) = -2.;

		cvm::srmatrix mY = mB.inv();
		std::cout << mY << (mB * mY * mB - mB).norm2() << std::endl;
	}
	catch (const cvm::cvmexception &e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// condition number
	std::cout << "\n>>> condition number" << std::endl;
	try
	{
		double a[] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
		cvm::srmatrix m(a, 3);
		std::cout << m.cond() << std::endl
			<< m.det() << std::endl << std::endl;
		m(3,3) = 10.;
		std::cout << m.cond() << std::endl << m.det() << std::endl;
	}
	catch (const cvm::cvmexception &e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// rank
	std::cout << "\n>>> rank" << std::endl;
	try
	{
		double a[] = { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. };
		cvm::rmatrix m(a, 3, 4);
		std::cout << m;
		std::cout << m.rank() << std::endl;

		m(3, 4) = 13.;
		std::cout << m.rank() << std::endl;
	}
	catch (const cvm::cvmexception &e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// normalize
	std::cout << "\n>>> normalize" << std::endl;
	try
	{
		double a[] = { 1., 2., 3., 4., 5., 6. };
		cvm::rmatrix m(a, 2, 3);

		m.normalize();
		std::cout << m;
	}
	catch (const cvm::cvmexception &e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// sum
	std::cout << "\n>>> sum" << std::endl;
	try
	{
		double a[] = { 1., 2., 3., 4., 5., 6. };
		const cvm::rmatrix m1(a, 2, 3);
		cvm::rmatrix m2(2, 3);
		cvm::rmatrix m(2, 3);
		m2.set(1.);

		std::cout << m.sum(m1, m2) << std::endl;
		std::cout << m.sum(m, m2);
	}
	catch (const cvm::cvmexception &e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// diff
	std::cout << "\n>>> diff" << std::endl;
	try
	{
		double a[] = { 1., 2., 3., 4., 5., 6. };
		const cvm::rmatrix m1(a, 2, 3);
		cvm::rmatrix m2(2, 3);
		cvm::rmatrix m(2, 3);
		m2.set(1.);

		std::cout << m.diff(m1, m2) << std::endl;
		std::cout << m.diff(m, m2);
	}
	catch (const cvm::cvmexception &e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// randomize
	std::cout << "\n>>> randomize" << std::endl;
	try
	{
		cvm::rmatrix m(3, 2);

		m.randomize(2., 5.);
		std::cout << m;
	}
	catch (const cvm::cvmexception &e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}
}

}  // namespace my_cvm
