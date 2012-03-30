#include <cvm/cvm.h>
#include <string>


namespace {
namespace local {

void svd_rmatrix()
{
	std::cout.setf(std::ios::scientific | std::ios::showpos);
	std::cout.precision(10);

	try
	{
		double m[] = {
			1., -1., 1., 2.,
			-2., 1., 3., -2.,
			1., 0., -2., 1.
		};

		cvm::rmatrix mA(m, 4, 3);  // column-major matrix
		cvm::rmatrix mS(4, 3);
		cvm::rvector vS(3);
		cvm::srmatrix mU(4), mVH(3);

		// A = U * S * V^H
		vS.svd(mA, mU, mVH);
		mS.diag(0) = vS;

		//std::cout << mA << std::endl;
		std::cout << mU << std::endl;
		std::cout << ~mVH << std::endl;
		std::cout << mS << std::endl;
		std::cout << (mA * ~mVH - mU * mS).norm() << std::endl;
		std::cout << (~mA * mU - ~(mS * mVH)).norm() << std::endl;
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}
}
	
}  // namespace local
}  // unnamed namespace

void svd()
{
	local::svd_rmatrix();
}
