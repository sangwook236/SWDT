//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <boost/math/constants/constants.hpp>
#include <algorithm>
#include <iostream>
#include <cmath>


namespace {
namespace local {

void compute_structure_tensor(const cv::Mat &img, const double sigma, cv::Mat &eigVal1, cv::Mat &eigVal2, cv::Mat &eigVec1, cv::Mat &eigVec2)
{
	const double sigma2 = sigma * sigma;
	const double _2sigma2 = 2.0 * sigma2;
	const double sigma3 = sigma2 * sigma;
	const double den = std::sqrt(2.0 * boost::math::constants::pi<double>()) * sigma3;

	const int kernel_size = 2 * (int)std::ceil(sigma) + 1;
	cv::Mat kernelX(1, kernel_size, CV_64FC1), kernelY(kernel_size, 1, CV_64FC1);

	for (int i = 0, k = -kernel_size/2; k <= kernel_size/2; ++i, ++k)
	{
		kernelX.at<double>(0, i) = k * std::exp(-k*k / _2sigma2) / den;
		kernelY.at<double>(i, 0) = k * std::exp(-k*k / _2sigma2) / den;
	}

	cv::Mat img_double;
	img.convertTo(img_double, CV_64FC1, 1.0 / 255.0, 0.0);
	cv::Mat Ix, Iy;
	cv::filter2D(img_double, Ix, -1, kernelX, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	cv::filter2D(img_double, Iy, -1, kernelY, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

	const cv::Mat Ix2 = Ix.mul(Ix);
	const cv::Mat Iy2 = Iy.mul(Iy);
	const cv::Mat IxIy = Ix.mul(Iy);

	// TODO [add] >> if Gaussian blur is required, blurring is applied to Ix2, Iy2, & IxIy.

	// structure tensor at point (i, j), S = [ Ix2(i, j) IxIy(i, j) ; IxIy(i, j) Iy2(i, j) ];
	const cv::Mat detS = Ix2.mul(Iy2) - IxIy.mul(IxIy);
	const cv::Mat S11_plus_S22 = Ix2 + Iy2;
	cv::Mat sqrtDiscriminant;
	cv::sqrt(S11_plus_S22.mul(S11_plus_S22) - 4.0 * detS, sqrtDiscriminant);

	// eigenvalues
	eigVal1 = (S11_plus_S22 + sqrtDiscriminant) * 0.5;
	eigVal2 = (S11_plus_S22 - sqrtDiscriminant) * 0.5;
	// eigenvectors
	eigVec1 = cv::Mat::zeros(img.size(), CV_64FC2);
	eigVec2 = cv::Mat::zeros(img.size(), CV_64FC2);

	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			if (std::fabs(eigVal1.at<double>(i, j)) < std::fabs(eigVal2.at<double>(i, j)))
				std::swap(eigVal1.at<double>(i, j), eigVal2.at<double>(i, j));

			const double a = Ix2.at<double>(i, j);
			const double b = IxIy.at<double>(i, j);
			const double lambda1 = eigVal1.at<double>(i, j);
			const double lambda2 = eigVal2.at<double>(i, j);
			eigVec1.at<cv::Vec2d>(i, j) = cv::Vec2d(-b, a - lambda1);
			eigVec2.at<cv::Vec2d>(i, j) = cv::Vec2d(-b, a - lambda2);
		}
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void structure_tensor()
{
	//const std::string input_filename("./machine_vision_data/opencv/thinning_img_1.png");
	//const std::string input_filename("./machine_vision_data/opencv/thinning_img_2.jpg");
	const std::string input_filename("D:/working_copy/swl_https/cpp/bin/data/kinect_segmentation/kinect_depth_rectified_valid_20130614T162309.png");
	const cv::Mat &src = cv::imread(input_filename);
	if (src.empty())
	{
		std::cerr << "file not found: " << input_filename << std::endl;
		return;
	}

	cv::imshow("src image", src);

	{
		cv::Mat gray;
		cv::cvtColor(src, gray, CV_BGR2GRAY);

		const double sigma = 3.0;
		cv::Mat eigVal1, eigVal2, eigVec1, eigVec2;
		local::compute_structure_tensor(gray, sigma, eigVal1, eigVal2, eigVec1, eigVec2);

		// ratio of eigenvalues.
		cv::Mat evRatio;
		cv::Mat(eigVal1 / eigVal2).convertTo(evRatio, CV_32FC1, 1.0, 0.0);
		cv::imshow("structure tensor - result", evRatio);
	}

	cv::waitKey();

	cv::destroyAllWindows();
}

}  // namespace my_opencv
