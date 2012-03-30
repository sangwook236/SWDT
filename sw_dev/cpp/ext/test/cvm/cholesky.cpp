#include <cvm/cvm.h>
#include <string>


namespace {
namespace local {

void cholesky_srmatrix()
{
	try
	{
		double a[] = { 1., 2., 1., 2., 5., -1., 1., -1., 20. };
		const cvm::srsmatrix m(a, 3);
		cvm::srmatrix h(3);

		// A = U^T * U
		h.cholesky(m);

		std::cout << h << std::endl;
		std::cout << ~h * h - m;
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}
}

void cholesky_scmatrix()
{
	try
	{
		double r[] = { 1., 2., 1., 2., 15., -1., 1., -1., 20. };
		double i[] = { 0., -1., 2., 1., 0., 3., -2., -3., 0. };
		const cvm::schmatrix m(r, i, 3);
		cvm::scmatrix c(3);

		// A = U^T * U
		c.cholesky(m);

		std::cout << c << std::endl;
		std::cout << ~c * c - m;
	}
	catch (const cvm::cvmexception& e) 
	{
		std::cout << "Exception " << e.what() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void cholesky()
{
	local::cholesky_srmatrix();
	local::cholesky_scmatrix();
}
