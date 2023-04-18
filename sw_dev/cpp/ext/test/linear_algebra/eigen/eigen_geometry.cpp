//#include "stdafx.h"
#include <iostream>
//#define EIGEN2_SUPPORT 1
//#include <Eigen/Core>
#include <Eigen/Geometry>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_eigen {

// REF [site] >> https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html
void transformation_test()
{
	// Rotation.

	{
		const double angle = M_PI / 3.0;
		//Eigen::Rotation2D<double> R(angle);
		Eigen::Rotation2Dd R(angle);

		std::cout << "R.angle() = " << R.angle() << std::endl;
		std::cout << "R.smallestPositiveAngle() = " << R.smallestPositiveAngle() << std::endl;
		std::cout << "R.smallestAngle() = " << R.smallestAngle() << std::endl;

		std::cout << "R.inverse():\n" << R.inverse().toRotationMatrix() << std::endl;

		//R = Eigen::Rotation2Dd::Identity();
		R.fromRotationMatrix(Eigen::Rotation2Dd::Matrix2::Identity());
		std::cout << R.toRotationMatrix() << std::endl;

		// Spherical interpolation.
		//R.slerp();
	}

	{
		const double angle = M_PI / 3.0;
		const Eigen::Vector3d axis(0, 0, 1);

		//Eigen::Quaternion<double> q(std::cos(angle * 0.5), 0.0, 0.0, std::sin(angle * 0.5));
		Eigen::Quaterniond q(std::cos(angle * 0.5), 0.0, 0.0, std::sin(angle * 0.5));
		//Eigen::Quaterniond q(Eigen::Quaterniond::AngleAxisType(angle, axis));

		std::cout << "q.x() = " << q.x() << std::endl;
		std::cout << "q.y() = " << q.y() << std::endl;
		std::cout << "q.z() = " << q.z() << std::endl;
		std::cout << "q.w() = " << q.w() << std::endl;
		std::cout << "q.vec() = " << q.vec().transpose() << std::endl;
		std::cout << "q.coeffs() = " << q.coeffs().transpose() << std::endl;

		std::cout << "q.norm() = " << q.norm() << std::endl;
		std::cout << "q.squaredNorm() = " << q.squaredNorm() << std::endl;

		std::cout << "q.inverse() = " << q.inverse() << std::endl;
		std::cout << "q.conjugate() = " << q.conjugate() << std::endl;

		std::cout << "q.toRotationMatrix():\n" << q.toRotationMatrix() << std::endl;

		q = Eigen::Quaterniond::Identity();
		std::cout << q << std::endl;

		// Spherical interpolation.
		//q.slerp();
	}

	{
		const double angle = M_PI / 3.0;
		const Eigen::Vector3d axis(0, 0, 1);

		//Eigen::AngleAxis<double> R(angle, axis);
		Eigen::AngleAxisd R(angle, axis);
		//Eigen::AngleAxisd R(Eigen::Quaterniond(std::cos(angle * 0.5), 0.0, 0.0, std::sin(angle * 0.5)));

		std::cout << "R.angle() = " << R.angle() << std::endl;
		std::cout << "R.axis() = " << R.axis().transpose() << std::endl;

		std::cout << "R.inverse():\n" << R.inverse().toRotationMatrix() << std::endl;

		//R = Eigen::AngleAxisd::Identity();
		R.fromRotationMatrix(Eigen::AngleAxisd::Matrix3::Identity());
		std::cout << R.toRotationMatrix() << std::endl;
	}

	//-----
	// Scaling.

	{
		//Eigen::UniformScaling<double> S(2.0);
		Eigen::UniformScaling<double> S = Eigen::Scaling(2.0);

		std::cout << "S.factor() = " << S.factor() << std::endl;
		std::cout << "S.inverse() = " << S.inverse().factor() << std::endl;
	}

	{
		//Eigen::DiagonalMatrix<double, 2> S(2.0, 4.0);
		Eigen::DiagonalMatrix<double, 2> S = Eigen::Scaling(2.0, 4.0);

		std::cout << "S.diagonal() = " << S.diagonal().transpose() << std::endl;
		std::cout << "S.inverse() = " << S.inverse().diagonal().transpose() << std::endl;
	}

	{
		//Eigen::DiagonalMatrix<double, 3> S(2.0, 4.0, 8.0);
		Eigen::DiagonalMatrix<double, 3> S = Eigen::Scaling(2.0, 4.0, 8.0);

		std::cout << "S.diagonal() = " << S.diagonal().transpose() << std::endl;
		std::cout << "S.inverse() = " << S.inverse().diagonal().transpose() << std::endl;
	}

	{
		//Eigen::DiagonalMatrix<double, 5> S(2.0, 4.0, 8.0, 4.0, 2.0);
		Eigen::DiagonalMatrix<double, 5> S = Eigen::Scaling(Eigen::Vector<double, 5>(2.0, 4.0, 8.0, 4.0, 2.0));

		std::cout << "S.diagonal() = " << S.diagonal().transpose() << std::endl;
		std::cout << "S.inverse() = " << S.inverse().diagonal().transpose() << std::endl;
	}

	//-----
	// Translation.

	{
		//Eigen::Translation<double, 2> T(1.0, 2.0);
		Eigen::Translation2d T(1.0, 2.0);

		std::cout << "T.x() = " << T.x() << std::endl;
		std::cout << "T.y() = " << T.y() << std::endl;
		std::cout << "T.vector() = " << T.vector().transpose() << std::endl;
		std::cout << "T.translation() = " << T.translation().transpose() << std::endl;

		std::cout << "T.inverse() = " << T.inverse().vector().transpose() << std::endl;

		T = Eigen::Translation2d::Identity();
		std::cout << T.vector().transpose() << std::endl;
	}

	{
		//Eigen::Translation<double, 3> T(1.0, 2.0, 3.0);
		Eigen::Translation3d T(1.0, 2.0, 3.0);

		std::cout << "T.x() = " << T.x() << std::endl;
		std::cout << "T.y() = " << T.y() << std::endl;
		std::cout << "T.z() = " << T.z() << std::endl;
		std::cout << "T.vector() = " << T.vector().transpose() << std::endl;
		std::cout << "T.translation() = " << T.translation().transpose() << std::endl;

		std::cout << "T.inverse() = " << T.inverse().vector().transpose() << std::endl;

		T = Eigen::Translation3d::Identity();
		std::cout << T.vector().transpose() << std::endl;
	}

	{
		Eigen::Translation<double, 5> T(Eigen::Vector<double, 5>(1.0, 2.0, 3.0, 4.0, 5.0));

		std::cout << "T.x() = " << T.x() << std::endl;
		std::cout << "T.y() = " << T.y() << std::endl;
		std::cout << "T.z() = " << T.z() << std::endl;
		std::cout << "T.vector() = " << T.vector().transpose() << std::endl;
		std::cout << "T.translation() = " << T.translation().transpose() << std::endl;

		std::cout << "T.inverse() = " << T.inverse().vector().transpose() << std::endl;

		T = Eigen::Translation<double, 5>::Identity();
		std::cout << T.vector().transpose() << std::endl;
	}

	//-----
	// Affine transformation.

	{
		const Eigen::Vector3d translation(1, 2, 3);
		const double angle(M_PI / 3.0);
		const Eigen::Vector3d axis(0, 0, 1);
		const double scale = 2.0;

		//Eigen::Transform<double, 3, Eigen::Isometry> T = Eigen::Translation3d(translation) * Eigen::AngleAxisd(angle, axis);
		//Eigen::Isometry3d T = Eigen::Translation3d(translation) * Eigen::AngleAxisd(angle, axis);
		//Eigen::Transform<double, 3, Eigen::Affine> T = Eigen::Translation3d(translation) * Eigen::AngleAxisd(angle, axis) * Eigen::Scaling(scale);
		Eigen::Affine3d T = Eigen::Translation3d(translation) * Eigen::AngleAxisd(angle, axis) * Eigen::Scaling(scale);
		//Eigen::Transform<double, 3, Eigen::AffineCompact> T = Eigen::Translation3d(translation) * Eigen::AngleAxisd(angle, axis) * Eigen::Scaling(scale);
		//Eigen::AffineCompact3d T = Eigen::Translation3d(translation) * Eigen::AngleAxisd(angle, axis) * Eigen::Scaling(scale);
		//Eigen::Transform<double, 3, Eigen::Projective> T = Eigen::Translation3d(translation) * Eigen::AngleAxisd(angle, axis) * Eigen::Scaling(scale);
		//Eigen::Projective3d T = Eigen::Translation3d(translation) * Eigen::AngleAxisd(angle, axis) * Eigen::Scaling(scale);

		std::cout << "T.matrix():\n" << T.matrix() << std::endl;  // 4x4.
		std::cout << "T.linear():\n" << T.linear() << std::endl;  // 3x3.
		std::cout << "T.affine():\n" << T.affine() << std::endl;  // 3x3.
		std::cout << "T.translation() = " << T.translation().transpose() << std::endl;  // 3x1.
		std::cout << "T.rotation():\n" << T.rotation() << std::endl;  // 3x3.
		std::cout << "T.inverse():\n" << T.inverse().matrix() << std::endl;  // 4x4.

		T.makeAffine();
		std::cout << "T.makeAffine():\n" << T.matrix() << std::endl;

		const auto TT = Eigen::Transform<double, 3, Eigen::Affine>::Identity();
		std::cout << "TT.matrix():\n" << TT.matrix() << std::endl;
	}

	//-----
	// Linear transformation.

	{
		const float angle(M_PIf / 3.0f);
		const Eigen::Vector3f axis(0, 0, 1);
		const float scale = 2.0f;

		//Eigen::Matrix<float, 2, 2> M = Eigen::Rotation2Df(angle) * Eigen::Scaling(scale);
		//Eigen::Matrix2f M = Eigen::Rotation2Df(angle) * Eigen::Scaling(scale);
		//Eigen::Matrix<float, 3, 3> M = Eigen::AngleAxisf(angle, axis) * Eigen::Scaling(scale);
		Eigen::Matrix3f M = Eigen::AngleAxisf(angle, axis) * Eigen::Scaling(scale);

		std::cout << "M:\n" << M << std::endl;
	}
}

}  // namespace my_eigen
