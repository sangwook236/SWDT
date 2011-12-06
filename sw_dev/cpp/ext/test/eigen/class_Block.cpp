#include "stdafx.h"
#include <Eigen/Core>


USING_PART_OF_NAMESPACE_EIGEN

template<typename Derived>
Eigen::Block<Derived> topLeftCorner(MatrixBase<Derived> &m, int rows, int cols)
{
	return Eigen::Block<Derived>(m.derived(), 0, 0, rows, cols);
}

template<typename Derived>
const Eigen::Block<Derived> topLeftCorner(const MatrixBase<Derived> &m, int rows, int cols)
{
	return Eigen::Block<Derived>(m.derived(), 0, 0, rows, cols);
}

int dynamic_block(int, char **)
{
	Matrix4d m = Matrix4d::Identity();
	std::cout << topLeftCorner(4*m, 2, 3) << std::endl;  // calls the const version

	topLeftCorner(m, 2, 3) *= 5;  // calls the non-const version
	std::cout << "Now the matrix m is:" << std::endl << m << std::endl;

	return 0;
}
