//#include "stdafx.h"
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <string>
#include <iostream>
#include <cassert>
#if defined(WIN32)
#include <windows.h>
#endif


#define __KERNEL_SIZE 4

namespace my_opencv {

void print_opencv_matrix(const CvMat *mat);
void print_opencv_matrix(const cv::Mat &mat);

}  // namespace my_opencv

namespace {
namespace local {

void rotate_kernel_180_deg(const CvMat *src, CvMat *dst)
{
	for (int i = 0; i < src->height; ++i)
		for (int j = 0; j < src->width; ++j)
			cvSet2D(dst, dst->height - i - 1, dst->width - j - 1, cvGet2D(src, i, j));
}

template<typename T>
void correlation(const T *src, T *dst, const int src_width, const int src_height, const int dst_width, const int dst_height, const CvMat *kernel, const CvPoint &anchor)
{
	const int kwidth = kernel->width, kheight = kernel->height;

	for (int i = 0; i < src_height; ++i)
		for (int j = 0; j < src_width; ++j)
		{
			T corr = 0;
			for (int h = 0; h < kheight; ++h)
			{
				const int h1 = i + h - anchor.x;
				if (h1 < 0 || h1 >= src_height) continue;
				for (int w = 0; w < kwidth; ++w)
				{
					const int w1 = j + w - anchor.y;
					if (w1 < 0 || w1 >= src_width) continue;
					corr += (T)(cvGet2D(kernel, h, w).val[0] * src[h1*src_width + w1]);
				}
			}
			dst[i*dst_width + j] = corr;
		}
}

void correlation(const IplImage *src, IplImage *dst, const CvMat *kernel, const CvPoint &anchor = cvPoint(-1, -1))
{
	const int width = src->width, height = src->height;

	const CvPoint center = anchor.x < 0 || anchor.y < 0 ?
		cvPoint(kernel->height / 2 + kernel->height % 2 - 1, kernel->width / 2 + kernel->width % 2 - 1) : anchor;

	if (src->depth == IPL_DEPTH_8U)
	{
		typedef unsigned char element_type;
		correlation((const element_type*)src->imageData, (element_type*)dst->imageData, width, height, dst->width, dst->height, kernel, center);
	}
	else if (src->depth == IPL_DEPTH_32F)
	{
		typedef float element_type;
		correlation((const element_type*)src->imageData, (element_type*)dst->imageData, width, height, dst->width, dst->height, kernel, center);
	}
	else assert(false);
}

void correlation(const CvMat *src, CvMat *dst, const CvMat *kernel, const CvPoint &anchor = cvPoint(-1, -1))
{
	const int width = src->width, height = src->height;

	const CvPoint center = anchor.x < 0 || anchor.y < 0 ?
		cvPoint(kernel->height / 2 + kernel->height % 2 - 1, kernel->width / 2 + kernel->width % 2 - 1) : anchor;

	const int type = CV_MAT_TYPE(src->type);
	if (type == CV_8UC1)
		correlation(src->data.ptr, dst->data.ptr, width, height, dst->width, dst->height, kernel, center);
	else if (type == CV_32FC1)
		correlation(src->data.fl, dst->data.fl, width, height, dst->width, dst->height, kernel, center);
	else if (type == CV_64FC1)
		correlation(src->data.db, dst->data.db, width, height, dst->width, dst->height, kernel, center);
	else assert(false);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void convolution_correlation()
{
#if defined(__KERNEL_SIZE) && __KERNEL_SIZE == 2
	// OOPS !!!
	// results of correlation & convolution using a kernel with size 2 in OpenCV are different from those in matlab.
	// the reason is that a center of kernel in doing correlation & convolution is (1, 1) in OpenCV & (0, 0) in matlab (zero-based index)
	// the solution is to use filter anchor in OpenCV
	//const CvPoint filterAnchor(cvPoint(-1, -1));  // if it is at the kernel center
	const CvPoint filterAnchor(cvPoint(0, 0));  // the result is the same as one using replication in matlab
	const int dimKernel = 2;
	const double H[dimKernel * dimKernel] = { 1.0, 2.0, 3.0, 4.0 };
#elif defined(__KERNEL_SIZE) && __KERNEL_SIZE == 3
	// results of correlation & convolution using a kernel with size 3 in OpenCV are the same as those in matlab.
	// a center of kernel in doing correlation & convolution is (1, 1) in OpenCV & matlab (zero-based index)
	const CvPoint filterAnchor(cvPoint(-1, -1));  // if it is at the kernel center
	const int dimKernel = 3;
	const double H[dimKernel * dimKernel] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
#elif defined(__KERNEL_SIZE) && __KERNEL_SIZE == 4
	// OOPS !!!
	// results of correlation & convolution using a kernel with size 2 in OpenCV are different from those in matlab.
	// the reason is that a center of kernel in doing correlation & convolution is (2, 2) in OpenCV & (1, 1) in matlab (zero-based index)
	// the solution is to use filter anchor in OpenCV
	//const CvPoint filterAnchor(cvPoint(-1, -1));  // if it is at the kernel center
	const CvPoint filterAnchor(cvPoint(1, 1));  // the result is the same as one using replication in matlab
	const int dimKernel = 4;
	const double H[dimKernel * dimKernel] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
#endif

	CvMat matH = cvMat(dimKernel, dimKernel, CV_64FC1, (void *)H);  // caution !!!: row-major matrix
	CvMat *matH_180 = cvCreateMat(dimKernel, dimKernel, CV_64FC1);
	local::rotate_kernel_180_deg(&matH, matH_180);

	//const int iplDepth = IPL_DEPTH_8U;
	const int iplDepth = IPL_DEPTH_32F;

	const int rdim = 4, cdim = 4;
	IplImage *imgA = cvCreateImage(cvSize(rdim, cdim), iplDepth, 1);
	IplImage *imgA_corr = cvCreateImage(cvSize(rdim, cdim), iplDepth, 1);
	IplImage *imgA_conv = cvCreateImage(cvSize(rdim, cdim), iplDepth, 1);

	if (iplDepth == IPL_DEPTH_8U)
	{
		for (int i = 0, ii = 1; i < rdim; ++i)
			for (int j = 0; j < cdim; ++j, ++ii)
				imgA->imageData[i*cdim + j] = (unsigned char)ii;

		// correlation in OpenCV
		//   by border replication, not zero-padding
		// correlation in matlab
		//   filter2(H, A, 'same') or xcorr2(A, H) or imfilter(A, H, 0, 'same', 'corr'), by zero-padding
		//   imfilter(A, H, 'replicate', 'same', 'corr'), by border replication
		//cvFilter2D(imgA, imgA_corr, &matH, filterAnchor);  // by border replication
		local::correlation(imgA, imgA_corr, &matH, filterAnchor);  // by zero-padding
		// convolution in OpenCV
		//   a correlation with kernel rotated 180 deg, by border replication, not zero-padding
		// convolution in matlab
		//   conv2(A, H) or imfilter(A, H, 0, 'same', 'conv'), by zero-padding
		//   imfilter(A, H, 'replicate', 'same', 'conv'), by border replication
		//cvFilter2D(imgA, imgA_conv, matH_180, filterAnchor);  // by border replication
		local::correlation(imgA, imgA_conv, matH_180, filterAnchor);  // by zero-padding

		std::cout << "source image =>" << std::endl;
		print_opencv_matrix(cv::Mat(imgA));
		std::cout << "kernel =>" << std::endl;
		print_opencv_matrix(&matH);
		std::cout << "filtered image: correlation =>" << std::endl;
		print_opencv_matrix(cv::Mat(imgA_corr));
		std::cout << "filtered image: convolution =>" << std::endl;
		print_opencv_matrix(cv::Mat(imgA_conv));
	}
	else if (iplDepth == IPL_DEPTH_32F)
	{
		float* p = (float*)imgA->imageData;
		for (int i = 0, ii = 1; i < rdim; ++i)
			for (int j = 0; j < cdim; ++j, ++ii)
				p[i*cdim + j] = (float)ii;

		// correlation in OpenCV
		//   by border replication, not zero-padding
		// correlation in matlab
		//   filter2(H, A, 'same') or xcorr2(A, H) or imfilter(A, H, 0, 'same', 'corr'), by zero-padding
		//   imfilter(A, H, 'replicate', 'same', 'corr'), by border replication
		//cvFilter2D(imgA, imgA_corr, &matH, filterAnchor);  // by border replication
		local::correlation(imgA, imgA_corr, &matH, filterAnchor);  // by zero-padding
		// convolution in OpenCV
		//   a correlation with kernel rotated 180 deg, by border replication, not zero-padding
		// convolution in matlab
		//   conv2(A, H) or imfilter(A, H, 0, 'same', 'conv'), by zero-padding
		//   imfilter(A, H, 'replicate', 'same', 'conv'), by border replication
		//cvFilter2D(imgA, imgA_conv, matH_180, filterAnchor);  // by border replication
		local::correlation(imgA, imgA_conv, matH_180, filterAnchor);  // by zero-padding

		std::cout << "source image =>" << std::endl;
		print_opencv_matrix(cv::Mat(imgA));
		std::cout << "kernel =>" << std::endl;
		print_opencv_matrix(&matH);
		std::cout << "filtered image: correlation =>" << std::endl;
		print_opencv_matrix(cv::Mat(imgA_corr));
		std::cout << "filtered image: convolution =>" << std::endl;
		print_opencv_matrix(cv::Mat(imgA_conv));
	}
	else
		assert(false);

	cvReleaseMat(&matH_180);
	cvReleaseImage(&imgA);
	cvReleaseImage(&imgA_corr);
	cvReleaseImage(&imgA_conv);

	//
	std::cout << "correlation by row & column vectors =>" << std::endl;
	{
		CvMat *mask1 = cvCreateMat(2 * 1 + 1, 1, CV_64FC1);
		CvMat *mask2 = cvCreateMat(1, 2 * 1 + 1, CV_64FC1);
		cvSet(mask1, cvScalarAll(1), 0);
		cvSet(mask2, cvScalarAll(1), 0);
		IplImage *imgA1 = cvCreateImage(cvSize(3, 3), IPL_DEPTH_32F, 1);
		IplImage *imgA2 = cvCreateImage(cvSize(3, 3), IPL_DEPTH_32F, 1);
		IplImage *imgA3 = cvCreateImage(cvSize(3, 3), IPL_DEPTH_32F, 1);
		float *p = (float*)imgA1->imageData;
		for (int i = 0, ii = 1; i < 3; ++i)
			for (int j = 0; j < 3; ++j, ++ii)
				p[i*3 + j] = (float)ii;
		local::correlation(imgA1, imgA2, mask1);  // by zero-padding
		local::correlation(imgA2, imgA3, mask2);  // by zero-padding
		print_opencv_matrix(cv::Mat(imgA3));

		cvReleaseMat(&mask1);
		cvReleaseMat(&mask2);
		cvReleaseImage(&imgA1);
		cvReleaseImage(&imgA2);
		cvReleaseImage(&imgA3);
	}

	//
	std::cout << "correlation by row & column vectors =>" << std::endl;
	{
		CvMat *mask1 = cvCreateMat(2 * 1 + 1, 1, CV_64FC1);
		CvMat *mask2 = cvCreateMat(1, 2 * 1 + 1, CV_64FC1);
		cvSet(mask1, cvScalarAll(1), 0);
		cvSet(mask2, cvScalarAll(1), 0);
		CvMat *imgA1 = cvCreateMat(3, 3, CV_64FC1);
		CvMat *imgA2 = cvCreateMat(3, 3, CV_64FC1);
		CvMat *imgA3 = cvCreateMat(3, 3, CV_64FC1);
		double *p = (double*)imgA1->data.db;
		for (int i = 0, ii = 1; i < 3; ++i)
			for (int j = 0; j < 3; ++j, ++ii)
				p[i*3 + j] = (double)ii;
		local::correlation(imgA1, imgA2, mask1);  // by zero-padding
		local::correlation(imgA2, imgA3, mask2);  // by zero-padding
		print_opencv_matrix(cv::Mat(imgA3));

		cvReleaseMat(&mask1);
		cvReleaseMat(&mask2);
		cvReleaseMat(&imgA1);
		cvReleaseMat(&imgA2);
		cvReleaseMat(&imgA3);
	}
}

}  // namespace my_opencv
