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
	filenames.push_back("../data/machine_vision/opencv/pic1.png");
	filenames.push_back("../data/machine_vision/opencv/pic2.png");
	filenames.push_back("../data/machine_vision/opencv/pic3.png");
	filenames.push_back("../data/machine_vision/opencv/pic4.png");
	filenames.push_back("../data/machine_vision/opencv/pic5.png");
	filenames.push_back("../data/machine_vision/opencv/pic6.png");
	filenames.push_back("../data/machine_vision/opencv/stuff.jpg");
	filenames.push_back("../data/machine_vision/opencv/synthetic_face.png");
	filenames.push_back("../data/machine_vision/opencv/puzzle.png");
	filenames.push_back("../data/machine_vision/opencv/fruits.jpg");
	filenames.push_back("../data/machine_vision/opencv/lena_rgb.bmp");
	filenames.push_back("../data/machine_vision/opencv/hand_01.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_05.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_24.jpg");
#elif 0
	filenames.push_back("../data/machine_vision/opencv/hand_01.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_02.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_03.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_04.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_05.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_06.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_07.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_08.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_09.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_10.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_11.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_12.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_13.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_14.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_15.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_16.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_17.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_18.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_19.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_20.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_21.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_22.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_23.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_24.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_25.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_26.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_27.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_28.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_29.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_30.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_31.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_32.jpg");
#elif 0
	filenames.push_back("../data/machine_vision/opencv/hand_33.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_34.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_35.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_36.jpg");
#elif 1
	filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211659.png");
	filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211705.png");
	filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211713.png");
	filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211839.png");
	filenames.push_back("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211842.png");
#endif

	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		const cv::Mat img = cv::imread(*it, cv::IMREAD_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		cv::Mat gray;
		if (1 == img.channels())
			img.copyTo(gray);
		else
			cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
			//cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

		// smoothing
#if 0
		// METHOD #1: down-scale and up-scale the image to filter out the noise.

		{
			cv::Mat tmp;
			cv::pyrDown(gray, tmp);
			cv::pyrUp(tmp, gray);
		}
#elif 0
		// METHOD #2: Gaussian filtering.

		{
			// FIXME [adjust] >> adjust parameters.
			const int kernelSize = 3;
			const double sigma = 2.0;
			cv::GaussianBlur(gray, gray, cv::Size(kernelSize, kernelSize), sigma, sigma, cv::BORDER_DEFAULT);
		}
#elif 1
		// METHOD #3: box filtering.

		{
			// FIXME [adjust] >> adjust parameters.
			const int ddepth = -1;  // the output image depth. -1 to use src.depth().
			const int kernelSize = 3;
			const bool normalize = true;
			cv::boxFilter(gray.clone(), gray, ddepth, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), normalize, cv::BORDER_DEFAULT);
			//cv::blur(gray.clone(), gray, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), cv::BORDER_DEFAULT);  // use the normalized box filter.
		}
#elif 0
		// METHOD #4: bilateral filtering.

		{
			// FIXME [adjust] >> adjust parameters.
			const int diameter = -1;  // diameter of each pixel neighborhood that is used during filtering. if it is non-positive, it is computed from sigmaSpace.
			const double sigmaColor = 3.0;  // for range filter.
			const double sigmaSpace = 50.0;  // for space filter.
			cv::bilateralFilter(gray.clone(), gray, diameter, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);
		}
#else
		// METHOD #5: no filtering.

		//gray = gray;
#endif

		// compute x- & y-gradients.
		cv::Mat xgradient, ygradient;
#if 0
		// METHOD #1: using Sobel operator.

		{
			//const int ksize = 5;
			const int ksize = CV_SCHARR;  // use Scharr operator
			cv::Sobel(gray, xgradient, CV_64FC1, 1, 0, ksize, 1.0, 0.0, cv::BORDER_DEFAULT);
			cv::Sobel(gray, ygradient, CV_64FC1, 0, 1, ksize, 1.0, 0.0, cv::BORDER_DEFAULT);
		}
#elif 1
		// METHOD #2: using derivative of Gaussian distribution.

		{
			cv::Mat img_double;
			double minVal, maxVal;
			cv::minMaxLoc(gray, &minVal, &maxVal);
			gray.convertTo(img_double, CV_64FC1, 1.0 / (maxVal - minVal), -minVal / (maxVal - minVal));

			const double deriv_sigma = 3.0;
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

#if 0
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
			cv::imshow("image gradient - x-gradient", tmp);
			cv::minMaxLoc(Iy, &minVal, &maxVal);
			Iy.convertTo(tmp, CV_32FC1, 1.0 / maxVal, 0.0);
			cv::imshow("image gradient - y-gradient", tmp);
		}
#endif

		// magnitude & phase of gradient.
		cv::Mat gradient_mag, gradient_phase;
		cv::magnitude(xgradient, ygradient, gradient_mag);  // CV_64FC1.
		cv::phase(xgradient, ygradient, gradient_phase);  // CV_64FC1. [0, 2*pi].

		double minMag = 0.0, maxMag = 0.0;
		cv::minMaxLoc(gradient_mag, &minMag, &maxMag);
		double minPhase = 0.0, maxPhase = 0.0;
		cv::minMaxLoc(gradient_phase, &minPhase, &maxPhase);

		const double thresholdRatio = 0.05;
		const cv::Mat gradient_mask = gradient_mag >= (minMag + (maxMag - minMag) * thresholdRatio);
		//cv::Mat gradient_mask;
		//cv::compare(gradient_mag, minMag + (maxVal - minMag) * thresholdRatio, gradient_mask, cv::CMP_GT);

		// filtering.
		gradient_mag.setTo(cv::Scalar::all(0), 0 == gradient_mask);
		gradient_phase.setTo(cv::Scalar::all(0), 0 == gradient_mask);

		cv::Mat gradient_mag_img, gradient_phase_img;
		gradient_mag.convertTo(gradient_mag_img, CV_32FC1, 1.0 / (maxMag - minMag), -minMag / (maxMag - minMag));
		gradient_phase.convertTo(gradient_phase_img, CV_32FC1, 1.0 / (maxPhase - minPhase), -minPhase / (maxPhase - minPhase));

		cv::imshow("image gradient - input", img);
		cv::imshow("image gradient - magnitude of gradient", gradient_mag_img);
		cv::imshow("image gradient - phase of gradient", gradient_phase_img);
		cv::imshow("image gradient - gradient mask", gradient_mask);

#if 1
		// draw gradients.
		//cv::Mat gradient_rgb;
		//cv::cvtColor(gradient_mask, gradient_rgb, cv::COLOR_GRAY2BGR);
		cv::Mat gradient_rgb = img.clone();
		const double maxGradientLen = 20.0 / maxMag;
		for (int r = 0; r < gradient_mask.rows; ++r)
			for (int c = 0; c < gradient_mask.cols; ++c)
			{
				const unsigned char &pix = gradient_mask.at<unsigned char>(r, c);
				if (pix)
				{
					const double &dx = xgradient.at<double>(r, c) * maxGradientLen;
					const double &dy = ygradient.at<double>(r, c) * maxGradientLen;

					cv::line(gradient_rgb, cv::Point(c, r), cv::Point(cvRound(c + dx), cvRound(r + dy)), CV_RGB(255, 0, 0), 1, cv::LINE_8, 0);
					//cv::circle(gradient_rgb, cv::Point(c, r), 1, CV_RGB(0, 0, 255), cv::FILLED, cv::LINE_8, 0);
				}
			}

		cv::imshow("image gradient - gradient", gradient_rgb);
#endif

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyAllWindows();
}

}  // namespace my_opencv
