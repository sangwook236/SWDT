#include <cvm/cvm.h>
#include <string>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace cvm {

void vector_function()
{
	// max & min
	std::cout << ">>> max & min" << std::endl;
	try
	{
		double a[] = { 3., 2., -5., -4., 5., -6. };
		const cvm::rvector v(a, 6);

		std::cout << v;
		std::cout << v[v.indofmax()] << std::endl;  // caution !!!: maximum absolute value
		std::cout << v[v.indofmin()] << std::endl;  // caution !!!: minimum absolute value
		std::cout << *std::max_element(v.begin(), v.end()) << std::endl;
		std::cout << *std::min_element(v.begin(), v.end()) << std::endl;
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// index of max & min
	std::cout << ">>> index of max & min" << std::endl;
	try
	{
		double a[] = { 3., 2., -5., -4. };
		const cvm::rvector v(a, 4);

		std::cout << v;
		std::cout << v.indofmax() << std::endl;  // caution !!!: maximum absolute value
		std::cout << v.indofmin() << std::endl;  // caution !!!: minimum absolute value
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// norm
	std::cout << "\n>>> norm" << std::endl;
	try
	{
		double a[] = { 1., 2., 3. };
		const cvm::rvector v(a, 3);

		std::cout << v;
		std::cout << v.norm() << std::endl;
		std::cout << v.norminf() << std::endl;
		std::cout << v.norm1() << std::endl;
		std::cout << v.norm2() << std::endl;
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// normalize
	std::cout << "\n>>> normalize" << std::endl;
	try
	{
		cvm::rvector v(4);
		v(1) = 1.;
		v(2) = 2.;
		v(3) = 3.;
		v(4) = 4.;

		std::cout << v.normalize();
		std::cout << v.norm() << std::endl;
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// sum
	std::cout << "\n>>> sum" << std::endl;
	try
	{
		double a[] = { 1., 2., 3., 4. };
		double b[] = { 2., 3., 4., 5. };
		cvm::rvector va(a, 4);
		cvm::rvector vb(b, 4);
		cvm::rvector v(4);

		std::cout << v.sum(va, vb);
		std::cout << v.sum(v, va);
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// diff
	std::cout << "\n>>> diff" << std::endl;
	try
	{
		double a[] = { 1., 2., 3., 4. };
		double b[] = { 2., 3., 4., 5. };
		cvm::rvector va(a, 4);
		cvm::rvector vb(b, 4);
		cvm::rvector v(4);

		std::cout << v.diff(va, vb);
		std::cout << v.diff(v, va);
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}

	// randomize
	std::cout << "\n>>> randomize" << std::endl;
	try
	{
		cvm::rvector v(4);

		v.randomize(-2., 3.);
		std::cout << v;
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}
}

}  // namespace cvm
