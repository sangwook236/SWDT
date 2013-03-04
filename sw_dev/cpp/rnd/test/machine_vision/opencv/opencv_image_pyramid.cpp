//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <list>


namespace {
namespace local {

//#define __USE_THE_SAME_SIZE_PYRAMID 1

void gaussian_image_pyramid(const cv::Mat &img, const int depth, std::list<cv::Mat> &pyramids)
{
#if defined(__USE_THE_SAME_SIZE_PYRAMID)
	pyramids.push_back(img);

	// FIXME [check] >> I am not sure it's correct.
	//	it will be checked whether the size of Gaussian kernel is changed or not.
	cv::Mat img1 = img, img2, img2_exp;
	for (int d = 0; d < depth; ++d)
	{
		cv::pyrDown(img1, img2);  // reduce
		cv::pyrUp(img2, img2_exp);  // expand

		pyramids.push_back(img2_exp);

		img1 = img2_exp;
	}
#else
	pyramids.push_back(img);

	cv::Mat img1 = img, img2;
	for (int d = 0; d < depth; ++d)
	{
		cv::pyrDown(img1, img2);  // reduce
		//cv::pyrUp(img2, img2_exp);  // expand

		pyramids.push_back(img2);

		img1 = img2;
	}
#endif
}

void laplacian_image_pyramid(const cv::Mat &img, const int depth, std::list<cv::Mat> &pyramids)
// [ref] "The Laplacian Pyramid as a Compact Image Code" by Peter J. Burt AND Edward H. Adelson
// IEEE Trans. on Communication, 1983
{
	pyramids.push_back(img);

	cv::Mat img1 = img, img2, img2_exp;
	for (int d = 0; d < depth; ++d)
	{
		cv::pyrDown(img1, img2);  // reduce
		cv::pyrUp(img2, img2_exp);  // expand

		cv::Mat tmp;
		cv::equalizeHist(cv::Mat(img1 - img2_exp), tmp);
		//tmp = cv::Mat(img1 - img2_exp);
		pyramids.push_back(tmp);

#if defined(__USE_THE_SAME_SIZE_PYRAMID)
		// FIXME [check] >> I am not sure it's correct.
		//	it will be checked whether the size of Gaussian kernel is changed or not.
		img1 = img2_exp;
#else
		img1 = img2;
#endif
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void image_pyramid()
{
	const std::string filename("./machine_vision_data/opencv/lena_rgb.bmp");

	const cv::Mat &image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

	//
	const int PYRAMID_DEPTH = 4;

	std::list<cv::Mat> pyramids;
	//local::gaussian_image_pyramid(image, PYRAMID_DEPTH, pyramids);
	local::laplacian_image_pyramid(image, PYRAMID_DEPTH, pyramids);

	//
	const std::string windowName("image pyramid - input image");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	for (std::list<cv::Mat>::const_iterator it = pyramids.begin(); it != pyramids.end(); ++it)
		std::cout << it->rows << "," << it->cols << std::endl;

#if defined(__USE_THE_SAME_SIZE_PYRAMID)
	cv::Mat disp_img(image.rows, image.cols * pyramids.size(), image.type());
	int k = 0;
	for (std::list<cv::Mat>::const_iterator it = pyramids.begin(); it != pyramids.end(); ++it, ++k)
		it->copyTo(disp_img(cv::Range::all(), cv::Range(k * image.cols, (k+1) * image.cols)));
#else
	int len = 0;
	for (std::list<cv::Mat>::const_iterator it = pyramids.begin(); it != pyramids.end(); ++it)
		len += it->cols;

	cv::Mat disp_img(image.rows, len, image.type());
	int k = 0, start = 0;
	for (std::list<cv::Mat>::const_iterator it = pyramids.begin(); it != pyramids.end(); ++it, ++k)
	{
#if defined(__GNUC__)
        cv::Mat disp_img_roi(disp_img, cv::Range(0, it->rows), cv::Range(start, start + it->cols));
		it->copyTo(disp_img_roi);
#else
		it->copyTo(disp_img(cv::Range(0, it->rows), cv::Range(start, start + it->cols)));
#endif
		start += it->cols;
	}
#endif

	cv::imshow(windowName, disp_img);

	cv::waitKey(0);

	cv::destroyWindow(windowName);
}

}  // namespace my_opencv
