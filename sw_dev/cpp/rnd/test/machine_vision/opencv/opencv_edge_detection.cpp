//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace


namespace my_opencv {

void edge_detection()
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
#elif 1
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
	//filenames.push_back("../data/machine_vision/opencv/hand_33.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_34.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_35.jpg");
	//filenames.push_back("../data/machine_vision/opencv/hand_36.jpg");
#elif 0
	filenames.push_back("../data/machine_vision/opencv/simple_hand_01.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_02.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_03.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_04.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_05.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_06.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_07.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_08.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_09.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_10.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_11.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_12.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_13.jpg");
#elif 0
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
			std::cout << "image file not found: " << *it << std::endl;
			continue;
		}

		cv::Mat gray;
		if (1 == img.channels())
			img.copyTo(gray);
		else
			cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
			//cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

		// Smoothing.
#if 0
		// METHOD #1: Down-scale and up-scale the image to filter out the noise.

		{
			cv::Mat tmp;
			cv::pyrDown(gray, tmp);
			cv::pyrUp(tmp, gray);
		}
#elif 0
		// METHOD #2: Gaussian filtering.

		{
			// FIXME [adjust] >> Adjust parameters.
			const int kernelSize = 3;
			const double sigma = 2.0;
			cv::GaussianBlur(gray, gray, cv::Size(kernelSize, kernelSize), sigma, sigma, cv::BORDER_DEFAULT);
		}
#elif 0
		// METHOD #3: Box filtering.

		{
			// FIXME [adjust] >> Adjust parameters.
			const int ddepth = -1;  // The output image depth. -1 to use src.depth().
			const int kernelSize = 3;
			const bool normalize = true;
			cv::boxFilter(gray.clone(), gray, ddepth, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), normalize, cv::BORDER_DEFAULT);
			//cv::blur(gray.clone(), gray, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), cv::BORDER_DEFAULT);  // Use the normalized box filter.
		}
#elif 1
		// METHOD #4: Bilateral filtering.

		{
			// FIXME [adjust] >> Adjust parameters.
			const int diameter = -1;  // Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
			const double sigmaColor = 3.0;  // For range filter.
			const double sigmaSpace = 50.0;  // For space filter.
			cv::bilateralFilter(gray.clone(), gray, diameter, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);
		}
#else
		// METHOD #5: no filtering.

		//gray = gray;
#endif

#if false
		const int ddepth = CV_32FC1;
		const int ksize = 3;
		const double scale = 1.0;
		const double delta = 0.0;

		cv::Mat dx, dy;
		cv::Mat abs_dx, abs_dy;

		// Calculate the x and y gradients using Sobel operator.
		cv::Sobel(gray, dx, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
		cv::convertScaleAbs(dx, abs_dx);

		cv::Sobel(gray, dy, ddepth, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);
		cv::convertScaleAbs(dy, abs_dy);

		// Combine the two gradients.
		cv::Mat edge;
		cv::addWeighted(abs_dx, 0.5, abs_dy, 0.5, 0, edge);

		//cv::threshold(edge, edge, 100, 255, cv::THRESH_BINARY_INV);
#else
		// Run the edge detector on grayscale.
		const int lowerEdgeThreshold = 30, upperEdgeThreshold = 50;
		const bool useL2 = true;  // If true, use L2 norm. Otherwise, use L1 norm (faster).
		const int apertureSize = 3;  // Aperture size for the Sobel() operator.
		cv::Mat edge;
		cv::Canny(gray, edge, lowerEdgeThreshold, upperEdgeThreshold, apertureSize, useL2);

		//cv::threshold(edge, edge, 100, 255, cv::THRESH_BINARY);
#endif

#if 0
		// Don't need.

		// Thresholding.
		double minVal, maxVal;
		cv::minMaxLoc(edge, &minVal, &maxVal);

		const double threshold_ratio = 0.8;
		const double edgeThreshold = minVal + threshold_ratio * (maxVal - minVal);
		edge = edge >= edgeThreshold;
#endif

		//
		cv::Mat cedge;
		img.copyTo(cedge, edge);
		//cedge = cedge > 0;

		cv::imshow("edge detection - input", img);
		cv::imshow("edge detection - result", cedge);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyAllWindows();
}

}  // namespace my_opencv
