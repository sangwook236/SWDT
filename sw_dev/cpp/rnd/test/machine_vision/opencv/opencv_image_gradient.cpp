//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/math/constants/constants.hpp>
#include <iostream>
#include <list>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void image_gradient()
{
	std::list<std::string> filenames;
#if 0
	filenames.push_back("./machine_vision_data/opencv/pic1.png");
	filenames.push_back("./machine_vision_data/opencv/pic2.png");
	filenames.push_back("./machine_vision_data/opencv/pic3.png");
	filenames.push_back("./machine_vision_data/opencv/pic4.png");
	filenames.push_back("./machine_vision_data/opencv/pic5.png");
	filenames.push_back("./machine_vision_data/opencv/pic6.png");
	filenames.push_back("./machine_vision_data/opencv/stuff.jpg");
	filenames.push_back("./machine_vision_data/opencv/synthetic_face.png");
	filenames.push_back("./machine_vision_data/opencv/puzzle.png");
	filenames.push_back("./machine_vision_data/opencv/fruits.jpg");
	filenames.push_back("./machine_vision_data/opencv/lena_rgb.bmp");
	filenames.push_back("./machine_vision_data/opencv/hand_01.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_05.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_24.jpg");
#elif 1
	filenames.push_back("./machine_vision_data/opencv/hand_01.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_02.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_03.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_04.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_05.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_06.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_07.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_08.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_09.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_10.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_11.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_12.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_13.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_14.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_15.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_16.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_17.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_18.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_19.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_20.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_21.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_22.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_23.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_24.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_25.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_26.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_27.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_28.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_29.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_30.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_31.jpg");
	//filenames.push_back("./machine_vision_data/opencv/hand_32.jpg");
#elif 0
	filenames.push_back("./machine_vision_data/opencv/hand_33.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_34.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_35.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_36.jpg");
#endif

	const std::string windowName1("image gradient - original");
	const std::string windowName2("image gradient - processed");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	//
	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		const cv::Mat img = cv::imread(*it, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		cv::Mat gray;
		if (1 == img.channels())
			img.copyTo(gray);
		else
			cv::cvtColor(img, gray, CV_BGR2GRAY);
			//cv::cvtColor(img, gray, CV_RGB2GRAY);

		// compute x- & y-gradients.
		cv::Mat xgradient, ygradient;
#if 0
		// METHOD #1: using Sobel operator.

		//const int ksize = 5;
		const int ksize = CV_SCHARR;  // use Scharr operator
		cv::Sobel(gray, xgradient, CV_32FC1, 1, 0, ksize, 1.0, 0.0, cv::BORDER_DEFAULT);
		cv::Sobel(gray, ygradient, CV_32FC1, 0, 1, ksize, 1.0, 0.0, cv::BORDER_DEFAULT);
#elif 1
		// METHOD #2: using derivative of Gaussian distribution.

		{
			cv::Mat img_double;
			double minVal, maxVal;
			cv::minMaxLoc(gray, &minVal, &maxVal);
			gray.convertTo(img_double, CV_64FC1, 1.0 / (maxVal - minVal), -minVal / (maxVal - minVal));

			const double deriv_sigma = 3.0;
			const double blur_sigma = 2.0;

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
			cv::filter2D(img_double, xgradient, -1, kernelX, cv::Point(-1, -1), 0.0, cv::BORDER_DEFAULT);
			cv::filter2D(img_double, ygradient, -1, kernelY, cv::Point(-1, -1), 0.0, cv::BORDER_DEFAULT);
		}
#endif

#if 1
		// display gradients.
		{
			double minVal, maxVal;
			cv::minMaxLoc(xgradient, &minVal, &maxVal);
			std::cout << "x-gradient: min = " << minVal << ", max = " << maxVal << std::endl;
			cv::minMaxLoc(ygradient, &minVal, &maxVal);
			std::cout << "y-gradient: min = " << minVal << ", max = " << maxVal << std::endl;

			cv::Mat Ix = cv::abs(xgradient);
			cv::Mat Iy = cv::abs(ygradient);

			cv::Mat tmp;
			cv::minMaxLoc(Ix, &minVal, &maxVal);
			Ix.convertTo(tmp, CV_32FC1, 1.0 / maxVal, 0.0);
			cv::imshow("x-directional gradient - Sobel", tmp);
			cv::minMaxLoc(Iy, &minVal, &maxVal);
			Iy.convertTo(tmp, CV_32FC1, 1.0 / maxVal, 0.0);
			cv::imshow("y-directional gradient - Sobel", tmp);
		}
#endif

		cv::Mat gradient, gradient_mask;
		cv::magnitude(xgradient, ygradient, gradient);

		const double thresholdRatio = 0.1;
		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(gradient, &minVal, &maxVal);
		gradient_mask = gradient >= (minVal + (maxVal - minVal) * thresholdRatio);
		//cv::compare(gradient, minVal + (maxVal - minVal) * thresholdRatio, gradient_mask, cv::CMP_GT);

		cv::cvtColor(gradient_mask, gradient, CV_GRAY2BGR);

#if 0
		// draw gradients.
		const float maxGradientLen = 20.0f;
		for (int r = 0; r < gradient_mask.rows; ++r)
			for (int c = 0; c < gradient_mask.cols; ++c)
			{
				const unsigned char &pix = gradient_mask.at<unsigned char>(r, c);
				if (pix)
				{
					const float &dx = xgradient.at<float>(r, c) * maxGradientLen / (float)maxVal;
					const float &dy = ygradient.at<float>(r, c) * maxGradientLen / (float)maxVal;

					cv::line(gradient, cv::Point(c, r), cv::Point(cvRound(c + dx), cvRound(r + dy)), CV_RGB(255, 0, 0), 1, 8, 0);
					//cv::circle(gradient, cv::Point(c, r), 1, CV_RGB(0, 0, 255), CV_FILLED, 8, 0);
				}
			}
#endif

		//
		cv::imshow(windowName1, img);
		cv::imshow(windowName2, gradient);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyAllWindows();
}

}  // namespace my_opencv
