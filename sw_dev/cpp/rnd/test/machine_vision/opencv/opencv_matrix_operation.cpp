//#include "stdafx.h"
//#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <iostream>
#include <cassert>


namespace my_opencv {

void print_opencv_matrix(const CvMat* mat);

}  // namespace my_opencv

namespace {
namespace local {

void matrix_operation_1()
{
	// type, depth & channel
	std::cout << ">>> type, depth & channel" << std::endl;
	{
		CvMat *A = cvCreateMat(1, 1, CV_64FC1);

		const int type = CV_MAT_TYPE(A->type);
		std::cout << "type == CV_64FC1: " << (type == CV_64FC1) << std::endl;
		const int depth = CV_MAT_DEPTH(A->type);
		std::cout << "depth == CV_64F: " << (depth == CV_64F) << std::endl;
		const int channel = CV_MAT_CN(A->type);
		std::cout << "channel == 1: " << (channel == 1) << std::endl;

		cvReleaseMat(&A);
	}

	// from array
	std::cout << "\n>>> from array" << std::endl;
	{
		const int rdim = 3, cdim = 3;
		double arr[rdim * cdim] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

		CvMat A = cvMat(rdim, cdim, CV_64FC1, (void*)arr);  // caution !!!: row-major matrix
		//cvSetZero(&A);
		for (int i = 0; i < rdim; ++i)
			for (int j = 0; j < cdim; ++j)
				cvmSet(&A, i, j, i + j);
		my_opencv::print_opencv_matrix(&A);

		for (int i = 0; i < rdim * cdim; ++i)
			std::cout << arr[i] << ' ';
		std::cout << std::endl;
	}

	// clone matrix
	std::cout << "\n>>> clone matrix" << std::endl;
	{
		const int rdim = 3, cdim = 3;
		double arr[rdim * cdim] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

		CvMat A = cvMat(rdim, cdim, CV_64FC1, (void*)arr);  // caution !!!: row-major matrix
		CvMat *B = cvCloneMat(&A);

		my_opencv::print_opencv_matrix(B);

		cvReleaseMat(&B);
	}

	// submatrix, row, column
	std::cout << "\n>>> submatrix, row, column" << std::endl;
	{
		const int rdim = 3, cdim = 3;
		const int rdim2 = 2, cdim2 = 2;
		double arr[rdim * cdim] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
		CvMat A = cvMat(rdim, cdim, CV_64FC1, (void*)arr);  // caution !!!: row-major matrix
		CvMat *B = cvCreateMat(rdim2, cdim2, CV_64FC1);

		cvGetSubRect(&A, B, cvRect(1, 1, rdim2, cdim2));
		cvmSet(B, 0, 0, -100.0);
		std::cout << "submatrix =>" << std::endl;
		my_opencv::print_opencv_matrix(B);

		cvReleaseMat(&B);

		std::cout << "original matrix =>" << std::endl;
		my_opencv::print_opencv_matrix(&A);
/*
		cvGetRow(&A, submat, rowIdx);
		cvGetRows(&A, submat, startRowIdx, endRowIdx, deltaRow);
		cvGetCol(&A, submat, colIdx);
		cvGetCols(&A, submat, startColIdx, endColIdx, deltaCol);
*/
	}

	// matrix multiplication
	std::cout << "\n>>> matrix multiplication" << std::endl;
	{
		const int rdim = 3, cdim = 3;

		double arr[rdim * cdim] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };

		CvMat* A = cvCreateMat(rdim, cdim, CV_64FC1);
		CvMat B = cvMat(rdim, cdim, CV_64FC1, (void*)arr);
		CvMat* C = cvCreateMat(rdim, cdim, CV_64FC1);

		cvSetZero(A);
		for (int i = 0; i < rdim; ++i)
			for (int j = 0; j < cdim; ++j)
				cvmSet(A, i, j, (i+1) * (j+1) + i);

		// D = alpha*op(A)*op(B) + beta*op(C), where op(X) is X or X^T
		//   cvGEMM(A, B, alpha, C, beta, D, CV_GEMM_A_T | CV_GEMM_B_T | CV_GEMM_C_T)
		//cvGEMM(A, &B, 1.0, NULL, 0.0, C, 0);
		//cvMatMulAdd(A, &B, C, D);
		cvMatMul(A, &B, C);

		std::cout << "C = A * B =>" << std::endl;
		my_opencv::print_opencv_matrix(C);

		cvReleaseMat(&A);
		cvReleaseMat(&C);
	}

	// transpose & flip
	std::cout << "\n>>> transpose & flip" << std::endl;
	{
		const int rdim = 3, cdim = 3;

		CvMat* A = cvCreateMat(rdim, cdim, CV_64FC1);
		CvMat* B = cvCreateMat(rdim, cdim, CV_64FC1);
		cvSetIdentity(A);
		for (int i = 0; i < rdim; ++i)
			for (int j = 0; j < cdim; ++j)
				if (i != j) cvmSet(A, i, j, (i+1) * (j+1) + j);

		cvTranspose(A, B);
		//cvFlip(A, B, -1 or 0 or 1):

		std::cout << "A =" << std::endl;
		my_opencv::print_opencv_matrix(A);
		std::cout << "A^T =" << std::endl;
		my_opencv::print_opencv_matrix(B);

		cvReleaseMat(&A);
		cvReleaseMat(&B);
	}

	// det, trace, inversion, & solve
	std::cout << "\n>>> det, trace, inversion, & solve" << std::endl;
	{
		const int rdim = 3, cdim = 3;

		CvMat* A = cvCreateMat(rdim, cdim, CV_64FC1);
		CvMat* Ainv = cvCreateMat(rdim, cdim, CV_64FC1);
		CvMat* B = cvCreateMat(rdim, 1, CV_64FC1);
		CvMat* X = cvCreateMat(rdim, 1, CV_64FC1);

		cvSetIdentity(A);
		for (int i = 0; i < rdim; ++i)
			for (int j = 0; j < cdim; ++j)
				if (i != j) cvmSet(A, i, j, (i+1) * (j+1) + j);
		for (int i = 0; i < rdim; ++i)
			cvmSet(B, i, 0, i * 2.0);

		const double det = cvDet(A);
		const CvScalar tr = cvTrace(A);
		cvInvert(A, Ainv, CV_LU);  // CV_LU or CV_SVD or CV_SVD_SYM
		cvSolve(A, B, X, CV_SVD);  // CV_LU or CV_SVD or CV_SVD_SYM

		std::cout << "det(A) = " << det << std::endl;
		std::cout << "tr(A) = " << tr.val[0] << std::endl;
		std::cout << "A^-1 =" << std::endl;
		my_opencv::print_opencv_matrix(Ainv);
		std::cout << "A * X = B =>" << std::endl;
		my_opencv::print_opencv_matrix(X);

		cvReleaseMat(&A);
		cvReleaseMat(&Ainv);
		cvReleaseMat(&B);
		cvReleaseMat(&X);
	}

	// svd
	std::cout << "\n>>> svd" << std::endl;
	{
		const int rdim = 3, cdim = 3;
		CvMat* A = cvCreateMat(rdim, cdim, CV_64FC1);
		CvMat* U = cvCreateMat(rdim, cdim, CV_64FC1);
		CvMat* V = cvCreateMat(rdim, cdim, CV_64FC1);
		CvMat* W = cvCreateMat(rdim, cdim, CV_64FC1);
		CvMat* B = cvCreateMat(rdim, 1, CV_64FC1);
		CvMat* X = cvCreateMat(rdim, 1, CV_64FC1);

		cvSetIdentity(A);
		for (int i = 0; i < rdim; ++i)
			for (int j = 0; j < cdim; ++j)
				if (i != j) cvmSet(A, i, j, (i+1) * (j+1) + j);
		for (int i = 0; i < rdim; ++i)
			cvmSet(B, i, 0, i * 2.0);

		cvSVD(A, W, U, V, 0);  // 0 or CV_SVD_MODIFY_A or CV_SVD_U_T or CV_SVD_V_T
		cvSVBkSb(W, U, V, B, X, 0);  // 0 or CV_SVD_MODIFY_A or CV_SVD_U_T or CV_SVD_V_T

		std::cout << "U =" << std::endl;
		my_opencv::print_opencv_matrix(U);
		std::cout << "V =" << std::endl;
		my_opencv::print_opencv_matrix(V);
		std::cout << "W =" << std::endl;
		my_opencv::print_opencv_matrix(W);
		std::cout << "A * X = B =>" << std::endl;
		my_opencv::print_opencv_matrix(X);

		cvReleaseMat(&A);
		cvReleaseMat(&U);
		cvReleaseMat(&V);
		cvReleaseMat(&W);
		cvReleaseMat(&B);
		cvReleaseMat(&X);
	}

	// eigen
	std::cout << "\n>>> eigen" << std::endl;
	{
		const int rdim = 3, cdim = 3;
		CvMat* A = cvCreateMat(rdim, cdim, CV_64FC1);
		CvMat* V = cvCreateMat(rdim, cdim, CV_64FC1);
		CvMat* D = cvCreateMat(rdim, 1, CV_64FC1);

		cvSetIdentity(A);
		for (int i = 0; i < rdim; ++i)
			for (int j = 0; j < i; ++j)
			{
				cvmSet(A, i, j, (i+1) * (j+1));
				cvmSet(A, j, i, (i+1) * (j+1));
			}

		// computes eigenvalues and eigenvectors of symmetric matrix
		const double eps = DBL_EPSILON;
		cvEigenVV(A, V, D, eps);

		std::cout << "eigenvectors =" << std::endl;
		my_opencv::print_opencv_matrix(V);
		std::cout << "eigenvalues =" << std::endl;
		my_opencv::print_opencv_matrix(D);

		cvReleaseMat(&A);
		cvReleaseMat(&V);
		cvReleaseMat(&D);
	}
}

void matrix_operation_2()
{
	// element access
	std::cout << ">>> element access" << std::endl;
	{
		const int ROWS = 5, COLS = 7;

		cv::Mat mat(ROWS, COLS, CV_32FC1);
		for (int r = 0; r < ROWS; ++r)
			for (int c = 0; c < COLS; ++c)
				mat.at<float>(r, c) = float(r + 1);

		std::cout << "mat = " << mat << std::endl;
		std::cout << "mat(0:2,1:2) = " << mat(cv::Rect(1, 0, 2, 3)) << std::endl;
		std::cout << "mat(0:2,1:2) = " << mat(cv::Range(0, 3), cv::Range(1, 3)) << std::endl;

		const int winHalfSize = 1;
		cv::Mat meanMat(cv::Mat::zeros(ROWS, COLS, CV_32FC1));
		for (int r = winHalfSize; r < ROWS - winHalfSize; ++r)
			for (int c = winHalfSize; c < COLS - winHalfSize; ++c)
			{
				//meanMat.at<float>(r, c) = (float)cv::mean(mat(cv::Rect(c - winHalfSize, r - winHalfSize, 2 * winHalfSize + 1, 2 * winHalfSize + 1)))[0];
				meanMat.at<float>(r, c) = (float)cv::mean(mat(cv::Range(r - winHalfSize, r + winHalfSize + 1), cv::Range(c - winHalfSize, c + winHalfSize + 1)))[0];
			}

		std::cout << "mean(mat) = " << meanMat << std::endl;
	}

	// scalar multiplication
	std::cout << "\n>>> scalar multiplication" << std::endl;
	{
		cv::MatND mat1 = cv::Mat::ones(10, 10, CV_32F);
		cv::MatND mat2 = mat1 * 100.0;

		std::cout << "mat1 = " << mat1 << std::endl;
		std::cout << "mat2 = " << mat2 << std::endl;
	}

	// reshape
	std::cout << "\n>>> reshape" << std::endl;
	{
		// [  0:1,   2:3,   4:5,   6:7,   8:9
		//   10:11, 12:13, 14:15, 16:17, 18:19 ]
		const int arr[20] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
		const cv::Mat mat(2, 5, CV_32SC2, (void *)arr);  // row-major matrix

		std::cout << "mat = " << mat << std::endl;

		const cv::Vec2i &v = mat.at<cv::Vec2i>(1, 2);
		std::cout << "mat(1,2) = (" << v[0] << ',' << v[1] << ')' << std::endl;

		{
			const cv::Mat &mat1 = mat.reshape(0, 5);
			const cv::Mat &mat2 = mat.reshape(1, 5);

			std::cout << "size(mat1) = (" << mat1.rows << ',' << mat1.cols << ") ; channels(mat1) = " << mat1.channels() << std::endl;
			std::cout << "mat1 = " << mat1 << std::endl;
			std::cout << "size(mat2) = (" << mat2.rows << ',' << mat2.cols << ") ; channels(mat2) = " << mat2.channels() << std::endl;
			std::cout << "mat2 = " << mat2 << std::endl;
		}

		{
			const cv::Mat &row_vec = mat.reshape(0, 10);  // to row vector
			const cv::Mat &col_vec = mat.reshape(0, 1);  // to column vector

			std::cout << "row vector = " << row_vec << std::endl;
			std::cout << "column vector = " << col_vec << std::endl;
		}
	}

	// pseudo-inverse
	std::cout << "\n>>> pseudo-inverse" << std::endl;
	{
		const float arr[] = { 1, 2, 0, 3, 1, 3 };
		const cv::Mat mat(2, 3, CV_32FC1, (void *)arr);  // row-major matrix

		const cv::Mat &inv = mat.inv(cv::DECOMP_SVD);  // pseudo-inverse
		std::cout << "pinv(mat) = " << inv << std::endl;
	}

	// element-wise operation
	{
		const float arrA[] = { 1, 2, 3, 4, 5, 6 };
		const cv::Mat A(2, 3, CV_32FC1, (void *)arrA);  // row-major matrix
		const float arrB[] = { 2, 2, 2, 2, 2, 2 };
		const cv::Mat B(2, 3, CV_32FC1, (void *)arrB);  // row-major matrix

		std::cout << "A.mul(B) = " << A.mul(B) << std::endl;
		std::cout << "A / B = " << (A / B) << std::endl;

		const cv::Mat sqrtA(A.size(), CV_32FC1);
		cv::sqrt(A, sqrtA);
		std::cout << "sqrt(A) = " << sqrtA << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void matrix_operation()
{
	//local::matrix_operation_1();
	local::matrix_operation_2();
}

}  // namespace my_opencv
