#include <cvm/cvm.h>
#include <string>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void vector_operation()
{
	// from array
	std::cout << ">>> from array" << std::endl;
	try
	{
		double a[] = { 1., 2., 3., 4., 5., 6. };
		cvm::rvector v(a, 6);
		std::cout << v;

		v[2] *= -1.0;
		v[5] = 50.0;

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
		cvm::rvector v(5);

		//v = 2.0;  std::cout << v;  // error !!!
		std::cout << v.set(3.0);
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}
}
