#include <cvm/cvm.h>
#include <string>


namespace {
namespace local {

void qr_rmatrix()
{
	try
	{
		const cvm::treal a[] = { 1., 2., 3., 4., 5., 6. };
		const cvm::rmatrix mh(const_cast<cvm::treal*>(a), 2, 3);
		const cvm::rmatrix mv(const_cast<cvm::treal*>(a), 3, 2);
		cvm::srmatrix s2(2), s3(3);
		cvm::rmatrix h(2,3), v(3,2);

		// A = Q * R
		mh.qr(h, s3);
		std::cout << (cvm::eye_real(2) - ~cvm::rmatrix(h,1,1,2,2)*cvm::rmatrix(h,1,1,2,2)).norm()
			<< " " << (mh - h * s3).norm() << std::endl;

		mh.qr(s2, h);
		std::cout << (cvm::eye_real(2) - ~s2 * s2).norm()
			<< " " << (mh - s2 * h).norm() << std::endl;

		mv.qr(v, s2);
		std::cout << (cvm::eye_real(2) - ~v * v).norm()
			<< " " << (mv - v * s2).norm() << std::endl;

		mv.qr(s3, v);
		std::cout << (cvm::eye_real(3) - ~s3 * s3).norm()
			<< " " << (mv - s3 * v).norm() << std::endl;
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}
}

void qr_cmatrix()
{
	try
	{
		cvm::treal ar[] = {1., 2., 3., 4., 5., 6.};
		cvm::treal ai[] = {1., -1., 2., -2., 3., -3.};
		const cvm::cmatrix mh(ar, ai, 2, 3);
		const cvm::cmatrix mv(ar, ai, 3, 2);
		cvm::scmatrix s2(2), s3(3);
		cvm::cmatrix h(2,3), v(3,2);

		// A = Q * R
		mh.qr(h, s3);
		std::cout << (cvm::eye_complex(2)-~cvm::cmatrix(h,1,1,2,2)*cvm::cmatrix(h,1,1,2,2)).norm()
			<< " " << (mh - h * s3).norm() << std::endl;

		mh.qr(s2, h);
		std::cout << (cvm::eye_complex(2) - ~s2 * s2).norm()
			<< " " << (mh - s2 * h).norm() << std::endl;

		mv.qr(v, s2);
		std::cout << (cvm::eye_complex(2) - ~v * v).norm()
			<< " " << (mv - v * s2).norm() << std::endl;

		mv.qr(s3, v);
		std::cout << (cvm::eye_complex(3) - ~s3 * s3).norm()
			<< " " << (mv - s3 * v).norm() << std::endl;
	}
	catch (const cvm::cvmexception& e)
	{
		std::cout << "Exception " << e.what() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace cvm {

void qr()
{
	local::qr_rmatrix();
	local::qr_cmatrix();
}

}  // namespace cvm
