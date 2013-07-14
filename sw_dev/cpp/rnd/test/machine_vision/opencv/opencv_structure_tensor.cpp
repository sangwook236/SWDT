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

void structure_tensor_2d(const cv::Mat &img, const double deriv_sigma, const double blur_sigma, cv::Mat &eval1, cv::Mat &eval2, cv::Mat &evec1, cv::Mat &evec2)
{
	const double sigma2 = deriv_sigma * deriv_sigma;
	const double _2sigma2 = 2.0 * sigma2;
	const double sigma3 = sigma2 * deriv_sigma;
	const double den = std::sqrt(2.0 * boost::math::constants::pi<double>()) * sigma3;

	const int deriv_kernel_size = 2 * (int)std::ceil(deriv_sigma) + 1;
	cv::Mat kernelX(1, deriv_kernel_size, CV_64FC1), kernelY(deriv_kernel_size, 1, CV_64FC1);

	// construct derivative kernels.
	for (int i = 0, k = -deriv_kernel_size/2; k <= deriv_kernel_size/2; ++i, ++k)
	{
		const double val = k * std::exp(-k*k / _2sigma2) / den;
		kernelX.at<double>(0, i) = val;
		kernelY.at<double>(i, 0) = val;
	}

	// compute x- & y-gradients.
	cv::Mat Ix, Iy;
	cv::filter2D(img, Ix, -1, kernelX, cv::Point(-1, -1), 0.0, cv::BORDER_DEFAULT);
	cv::filter2D(img, Iy, -1, kernelY, cv::Point(-1, -1), 0.0, cv::BORDER_DEFAULT);

	// solve eigensystem.

	const cv::Mat Ix2 = Ix.mul(Ix);  // Ix^2 = Ix * Ix
	const cv::Mat Iy2 = Iy.mul(Iy);  // Iy^2 = Iy * Iy
	const cv::Mat IxIy = Ix.mul(Iy);  // Ix * Iy

#if 1
	// TODO [add] >> if Gaussian blur is required, blurring is applied to Ix2, Iy2, & IxIy.
	const int blur_kernel_size = 2 * (int)std::ceil(blur_sigma) + 1;
	cv::GaussianBlur(Ix2, Ix2, cv::Size(blur_kernel_size, blur_kernel_size), blur_sigma, blur_sigma, cv::BORDER_DEFAULT);
	cv::GaussianBlur(Iy2, Iy2, cv::Size(blur_kernel_size, blur_kernel_size), blur_sigma, blur_sigma, cv::BORDER_DEFAULT);
	cv::GaussianBlur(IxIy, IxIy, cv::Size(blur_kernel_size, blur_kernel_size), blur_sigma, blur_sigma, cv::BORDER_DEFAULT);
#endif

	// structure tensor at point (i, j), S = [ Ix2(i, j) IxIy(i, j) ; IxIy(i, j) Iy2(i, j) ];
	const cv::Mat detS = Ix2.mul(Iy2) - IxIy.mul(IxIy);
	const cv::Mat S11_plus_S22 = Ix2 + Iy2;
#if 0
	cv::Mat sqrtDiscriminant(img.size(), CV_64FC1);
	cv::sqrt(S11_plus_S22.mul(S11_plus_S22) - 4.0 * detS, sqrtDiscriminant);
#else
	cv::Mat sqrtDiscriminant(S11_plus_S22.mul(S11_plus_S22) - 4.0 * detS);

	const double tol = 1.0e-10;
	const int count1 = cv::countNonZero(sqrtDiscriminant < 0.0);
	if (count1 > 0)
	{
		std::cout << "non-zero count = " << count1 << std::endl;

		const int count2 = cv::countNonZero(sqrtDiscriminant < -tol);
		if (count2 > 0)
		{
#if defined(DEBUG) || defined(_DEBUG)
			for (int i = 0; i < img.rows; ++i)
				for (int j = 0; j < img.cols; ++j)
					if (sqrtDiscriminant.at<double>(i, j) < 0.0)
						std::cout << i << ", " << j << " = " << sqrtDiscriminant.at<double>(i, j) << std::endl;
#endif

			std::cerr << "complex eigenvalues exist" << std::endl;
			return;
		}
		else
			sqrtDiscriminant.setTo(0.0, sqrtDiscriminant < 0.0);
	}

	cv::sqrt(sqrtDiscriminant, sqrtDiscriminant);
#endif

	// eigenvalues
	eval1 = (S11_plus_S22 + sqrtDiscriminant) * 0.5;
	eval2 = (S11_plus_S22 - sqrtDiscriminant) * 0.5;
	// eigenvectors
	evec1 = cv::Mat::zeros(img.size(), CV_64FC2);
	evec2 = cv::Mat::zeros(img.size(), CV_64FC2);

	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			if (std::fabs(eval1.at<double>(i, j)) < std::fabs(eval2.at<double>(i, j)))
				std::swap(eval1.at<double>(i, j), eval2.at<double>(i, j));

			const double a = Ix2.at<double>(i, j);
			const double b = IxIy.at<double>(i, j);
			const double lambda1 = eval1.at<double>(i, j);
			const double lambda2 = eval2.at<double>(i, j);
			evec1.at<cv::Vec2d>(i, j) = cv::Vec2d(-b, a - lambda1);
			evec2.at<cv::Vec2d>(i, j) = cv::Vec2d(-b, a - lambda2);
		}
}

void compute_valid_region_using_coherence(const cv::Mat &eval1, const cv::Mat &eval2, const cv::Mat &valid_eval_region_mask, const cv::Mat &constant_region_mask, cv::Mat &valid_region)
{
	// coherence = 1 when the gradient is totally aligned, and coherence = 0 (lambda1 = lambda2) when it has no predominant direction.
	cv::Mat coherence((eval1 - eval2) / (eval1 + eval2));  // if eigenvalue2 > 0.
	coherence = coherence.mul(coherence);

	double minVal, maxVal;
	cv::minMaxLoc(coherence, &minVal, &maxVal);
	std::cout << "coherence: min = " << minVal << ", max = " << maxVal << std::endl;

#if 0
	const double threshold = 0.5;
	valid_region = coherence <= threshold;
#elif 0
	const double threshold = 0.9;
	valid_region = coherence >= threshold;
#else
	const double threshold1 = 0.2, threshold2 = 0.8;
	valid_region = threshold1 <= coherence & coherence <= threshold2;
#endif

	valid_region.setTo(cv::Scalar::all(0), constant_region_mask);
}

void compute_valid_region_using_ev_ratio(const cv::Mat &eval1, const cv::Mat &eval2, const cv::Mat &valid_eval_region_mask, const cv::Mat &constant_region_mask, cv::Mat &valid_region)
{
	cv::Mat eval_ratio(valid_eval_region_mask.size(), CV_8UC1, cv::Scalar::all(0));
	cv::Mat(eval1 / eval2).copyTo(eval_ratio, valid_eval_region_mask);

	double minVal, maxVal;
	cv::minMaxLoc(eval_ratio, &minVal, &maxVal);
	std::cout << "ev ratio: min = " << minVal << ", max = " << maxVal << std::endl;

#if 0
	const double threshold = 0.5;
	valid_region = cv::abs(eval_ratio - 1.0f) <= threshold;  // if lambda1 = lambda2, the gradient in the window has no predominant direction.
#else
	const double threshold1 = 1.0, threshold2 = 5.0;
	valid_region = threshold1 <= eval_ratio & eval_ratio <= threshold2;
#endif

	valid_region.setTo(cv::Scalar::all(0), constant_region_mask);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void structure_tensor()
{
#if 1
	const std::string input_filename("./machine_vision_data/opencv/synthesized_training_image1.bmp");
	//const std::string input_filename("./machine_vision_data/opencv/thinning_img_1.png");
	//const std::string input_filename("./machine_vision_data/opencv/thinning_img_2.jpg");
	const cv::Mat &src = cv::imread(input_filename);

	const bool useGray = true;
#elif 0
	//const std::string input_filename("D:/working_copy/swl_https/cpp/bin/data/kinect_segmentation/kinect_depth_rectified_valid_20130614T162309.png");
	//const std::string input_filename("D:/working_copy/swl_https/cpp/bin/data/kinect_segmentation/kinect_depth_rectified_valid_20130614T162314.png");
	//const std::string input_filename("D:/working_copy/swl_https/cpp/bin/data/kinect_segmentation/kinect_depth_rectified_valid_20130614T162348.png");
	const std::string input_filename("D:/working_copy/swl_https/cpp/bin/data/kinect_segmentation/kinect_depth_rectified_valid_20130614T162459.png");
	//const std::string input_filename("D:/working_copy/swl_https/cpp/bin/data/kinect_segmentation/kinect_depth_rectified_valid_20130614T162525.png");
	//const std::string input_filename("D:/working_copy/swl_https/cpp/bin/data/kinect_segmentation/kinect_depth_rectified_valid_20130614T162552.png");
	const cv::Mat &src = cv::imread(input_filename, cv::IMREAD_UNCHANGED);

	const bool useGray = false;
#endif
	if (src.empty())
	{
		std::cerr << "file not found: " << input_filename << std::endl;
		return;
	}

	{
		cv::Mat img_double;
		double minVal, maxVal;

		if (useGray)
		{
			cv::Mat gray;
			cv::cvtColor(src, gray, CV_BGR2GRAY);

			gray.convertTo(img_double, CV_64FC1, 1.0 / 255.0, 0.0);
		}
		else
		{
			cv::minMaxLoc(src, &minVal, &maxVal);
			src.convertTo(img_double, CV_64FC1, 1.0 / (maxVal - minVal), -minVal / (maxVal - minVal));
		}

		{
			cv::Mat tmp;
			img_double.convertTo(tmp, CV_32FC1, 1.0, 0.0);
			cv::imshow("src image", tmp);
		}

		const double deriv_sigma = 3.0;
		const double blur_sigma = 2.0;
		cv::Mat eval1, eval2, evec1, evec2;
		{
			boost::timer::auto_cpu_timer timer;
			local::structure_tensor_2d(img_double, deriv_sigma, blur_sigma, eval1, eval2, evec1, evec2);
		}

		// post-processing.
		eval1 = cv::abs(eval1);
		eval2 = cv::abs(eval2);

		cv::minMaxLoc(eval1, &minVal, &maxVal);
		std::cout << "max eigenvalue: " << minVal << ", " << maxVal << std::endl;
		cv::minMaxLoc(eval2, &minVal, &maxVal);
		std::cout << "min eigenvalue: " << minVal << ", " << maxVal << std::endl;

		const double tol = 1.0e-10;
		const cv::Mat valid_eval_region_mask(eval2 >= tol);
		const cv::Mat constant_region_mask(eval1 < tol & eval2 < tol);  // if lambda1 = lambda2 = 0, the image within the window is constant.

		// METHOD #1; using coherence.
		//	[ref] http://en.wikipedia.org/wiki/Structure_tensor
		{
			cv::Mat valid_region;
			{
				boost::timer::auto_cpu_timer timer;
				local::compute_valid_region_using_coherence(eval1, eval2, valid_eval_region_mask, constant_region_mask, valid_region);
			}

			cv::imshow("structure tensor - coherence", valid_region);
			//cv::imwrite("./machine_vision_data/opencv/structure_tensor_coherence.png", valid_region);
		}

		// METHOD #2: using the ratio of eigenvales.
		{
			cv::Mat valid_region;
			{
				boost::timer::auto_cpu_timer timer;
				local::compute_valid_region_using_ev_ratio(eval1, eval2, valid_eval_region_mask, constant_region_mask, valid_region);
			}

			cv::imshow("structure tensor - ratio of eigenvalues", valid_region);
			//cv::imwrite("./machine_vision_data/opencv/structure_tensor_ev_ratio.png", valid_region);
		}
	}

	cv::waitKey();

	cv::destroyAllWindows();
}

}  // namespace my_opencv
