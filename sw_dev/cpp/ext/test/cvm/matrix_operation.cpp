#include <cvm/cvm.h>
#include <string>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

void matrix_operation()
{
	// from array
	std::cout << ">>> from array" << std::endl;
	try
	{
		double a[] = { 1., 2., 3., 4., 5., 6. };
		cvm::rmatrix m(a, 2, 3);  // caution !!!: column-major matrix
		std::cout << m;

		m[2] *= -1.0;  // return row vector
		//m(2) *= -1.0;  // return column vector
		m(1, 3) = 10.0;

		for (int i = 0; i < 6; ++i)
			std::cout << a[i] << ' ';
		std::cout << std::endl;
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// set values
	std::cout << "\n>>> set values" << std::endl;
	try
	{
		cvm::srmatrix m(3);

		//m = -12.3;  std::cout << m;  // error !!!
		std::cout << m.set(7.8);
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// set zeros
	std::cout << "\n>>> set zeros" << std::endl;
	try
	{
		cvm::srmatrix m(3);
		m.randomize(0., 1.);

		std::cout << m << std::endl;
		std::cout << m.vanish();
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// identity
	std::cout << "\n>>> identity" << std::endl;
	try
	{
		cvm::srmatrix m(3);
		m.randomize(0., 1.);

		std::cout << m << std::endl;
		std::cout << m.identity();
		//std::cout << eye_real(m);
		//std::cout << eye_complex(m);
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// transpose
	std::cout << "\n>>> transpose" << std::endl;
	try
	{
		double a[] = { 1., 2., 3., 4., 5., 6. };
		cvm::rmatrix m(a, 2, 3);  // column-major matrix
		cvm::rmatrix mt(3, 2);

		std::cout << m << std::endl << ~m << std::endl;

		mt.transpose(m);
		std::cout << mt << std::endl;

		mt.transpose();
		std::cout << mt;
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// diagnal
	std::cout << "\n>>> diagnal" << std::endl;
	try
	{
		double a[] = { 1., 2., 3., 4., 5., 6., 7., 8., 9. };
		cvm::rmatrix m(2, 3);
		const cvm::srmatrix ms(a, 3);

		// m.diag(i): i = 0 for main diagonal, i < 0 for loower diagonals and i > 0 for upper ones
		m.diag(-1).set(1.);
		m.diag(0).set(2.);
		m.diag(1).set(3.);
		m.diag(2).set(4.);

		std::cout << m << std::endl;
		std::cout << ms << std::endl;
		std::cout << ms.diag(0) << ms.diag(1);
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// submatrix
	std::cout << "\n>>> submatrix" << std::endl;
	try
	{
		cvm::rmatrix m(4, 5);
		cvm::rmatrix sm(m, 2, 2, 2, 3);
		sm.set(2.0);

		cvm::rmatrix m2(2, 2);
		for (int i = 1; i <= 2; ++i)
			for (int j = 1; j <= 2; ++j)
				m2(i, j) = -(i + j);
		m.assign(3, 4, m2);

		std::cout << m << std::endl;
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}
}
