//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>


namespace {
namespace local {

// REF [file] >> ${OPENCV_HOME}/samples/cpp/lsd_lines.cpp.
void lsd()
{
	const std::string img_filename("./data/feature_analysis/chairs.pgm");

	cv::Mat image = cv::imread(img_filename, cv::IMREAD_GRAYSCALE);
	if (image.empty())
	{
		std::cerr << "File not found: " << img_filename << std::endl;
		return;
	}

#if 0
	cv::Canny(image, image, 50, 200, 3);  // Apply canny edge.
#endif

#if 1
	const int kernelSize = 5;
	const double sigma = 3.0;
	cv::GaussianBlur(image, image, cv::Size(kernelSize, kernelSize), sigma, sigma, cv::BORDER_DEFAULT);
#endif

	// Create and LSD detector with standard or no refinement.
	//cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_NONE);
	cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
	//cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);

	const double start = double(cv::getTickCount());
	std::vector<cv::Vec4f> lines_std;

	// Detect the lines.
	ls->detect(image, lines_std);

	const double duration_ms = (double(cv::getTickCount()) - start) * 1000 / cv::getTickFrequency();
	std::cout << "It took " << duration_ms << " ms." << std::endl;

	// Show found lines.
	cv::Mat drawnLines(image);
	ls->drawSegments(drawnLines, lines_std);
	cv::imshow("Standard refinement", drawnLines);

	cv::waitKey();
	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void line()
{
	// Line segment detection (LSD).
	local::lsd();
}

}  // namespace my_opencv
