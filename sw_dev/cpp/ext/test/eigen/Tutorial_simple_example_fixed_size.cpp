//#include "stdafx.h"
#include <Eigen/Core>
#include <iostream>


// import most common Eigen types 
USING_PART_OF_NAMESPACE_EIGEN

namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

int fixed_size(int, char *[])
{
	Eigen::Matrix3f m3;
	m3 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
	Eigen::Matrix4f m4 = Eigen::Matrix4f::Identity();
	Eigen::Vector4i v4(1, 2, 3, 4);

	std::cout << "m3\n" << m3 << "\nm4:\n" << m4 << "\nv4:\n" << v4 << std::endl;

	return 0;
}
