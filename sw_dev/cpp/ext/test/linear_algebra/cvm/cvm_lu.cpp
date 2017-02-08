#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <cvm/cvm.h>
#else
#include <cvm.h>
#endif
#include <string>


namespace {
namespace local {

void lu_srmatrix()
{
	std::cout.setf(std::ios::scientific | std::ios::left);
	std::cout.precision(3);

	try
	{
		const cvm::treal a[] = { 1., -1., 1., 2., -2., 1., 3., -2., 1. };
		cvm::srmatrix ma(const_cast<cvm::treal*>(a), 3);
		cvm::srmatrix mLU(3);
		cvm::rmatrix mb1(3, 2); cvm::rvector vb1(3);
		cvm::rmatrix mb2(3, 2); cvm::rvector vb2(3);
		cvm::rmatrix mx1(3, 2); cvm::rvector vx1(3);
		cvm::rmatrix mx2(3, 2); cvm::rvector vx2(3);
		cvm::iarray nPivots(3);

		double dErr = 0.;
		mb1.randomize(-1.,3.); vb1.randomize(-2.,4.);
		mb2.randomize(-2.,5.); vb2.randomize(-3.,1.);

		// A = P * L * U
		mLU.low_up(ma, nPivots);

		mx1 = ma.solve_lu(mLU, nPivots, mb1, dErr);
		std::cout << mx1 << dErr << std::endl;

		mx2 = ma.solve_lu(mLU, nPivots, mb2);
		std::cout << mx2 << std::endl;;
		std::cout << ma * mx1 - mb1 << std::endl << ma * mx2 - mb2;

		vx1 = ma.solve_lu(mLU, nPivots, vb1, dErr);
		std::cout << vx1 << dErr << std::endl;

		vx2 = ma.solve_lu(mLU, nPivots, vb2);
		std::cout << vx2 << std::endl;;
		std::cout << ma * vx1 - vb1 << std::endl << ma * vx2 - vb2;
	}
	catch (const cvm::cvmexception &e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}
}

void lu_scmatrix()
{
	std::cout.setf(std::ios::scientific | std::ios::left);
	std::cout.precision(3);

	try
	{
		const cvm::treal a[] = { 1., 2., 3., 4., 5., 6., 7., 8., 10. };
		cvm::srmatrix m(const_cast<cvm::treal*>(a), 3);
		cvm::srmatrix mLU(3), mLo(3), mUp(3);
		cvm::iarray naPivots(3);

		// A = P * L * U
		mLU.low_up(m, naPivots);

		mLo.identity();
		mLo.diag(-2) = mLU.diag(-2);
		mLo.diag(-1) = mLU.diag(-1);
		mUp.diag(0) = mLU.diag(0);
		mUp.diag(1) = mLU.diag(1);
		mUp.diag(2) = mLU.diag(2);
		std::cout << mLo << std::endl << mUp << std::endl << naPivots << std::endl;

		mLU = mLo * mUp;
		for (int i = 3; i >= 1; --i)
			mLU.swap_rows(i, naPivots[i]);
		std::cout << mLU;
	}
	catch (const cvm::cvmexception &e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_cvm {

void lu()
{
	local::lu_srmatrix();
	local::lu_scmatrix();
}

}  // namespace my_cvm
