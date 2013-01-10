//#include "stdafx.h"
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <iostream>
#include <cassert>


namespace my_opencv {

void print_opencv_matrix(const CvMat *mat);

void vector_operation()
{
	// dot & cross
	std::cout << ">>> dot & cross" << std::endl;
	{
		const int dim = 3;

		const double arr1[dim] = { 1.0, 3.0, -2.0 };
		const double arr2[dim] = { 3.0, -1.0, 11.0 };

		CvMat v1 = cvMat(dim, 1, CV_64FC1, (void*)arr1);
		CvMat v2 = cvMat(dim, 1, CV_64FC1, (void*)arr2);
		CvMat* v = cvCreateMat(dim, 1, CV_64FC1);

		const double dot = cvDotProduct(&v1, &v2);
		cvCrossProduct(&v1, &v2, v);

		std::cout << "dot(v1, v2) = " << dot << std::endl;
		std::cout << "cross(v1, v2) =" << std::endl;
		print_opencv_matrix(v);

		cvReleaseMat(&v);
	}
}

}  // namespace my_opencv
