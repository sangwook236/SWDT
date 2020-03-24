//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/fast_line_detector.hpp>
#include <vector>
#include <iostream>


namespace {
namespace local {

// REF [file] >> ${OPENCV_HOME}/samples/cpp/lsd_lines.cpp.
void lsd()
{
	const std::string image_filepath("./data/feature_analysis/chairs.pgm");

	cv::Mat image(cv::imread(image_filepath, cv::IMREAD_GRAYSCALE));
	if (image.empty())
	{
		std::cerr << "File not found: " << image_filepath << std::endl;
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

	// Create an LSD detector with standard or no refinement.
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

// REF [site] >> https://docs.opencv.org/4.1.0/df/d4c/classcv_1_1ximgproc_1_1FastLineDetector.html
void fld()
{
	const std::string image_filepath("./data/feature_analysis/chairs.pgm");

	cv::Mat gray(cv::imread(image_filepath, cv::IMREAD_GRAYSCALE));
	if (gray.empty())
	{
		std::cerr << "File not found: " << image_filepath << std::endl;
		return;
	}

	// Create an LSD detector.
	cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector();

	// Create an FLD detector.
	const int length_threshold = 10;  // Segments shorter than this will be discarded.
	const float distance_threshold = 1.41421356f;  // A point placed from a hypothesis line segment farther than this will be regarded as an outlier.
	const double canny_th1 = 50.0;  // First threshold for hysteresis procedure in Canny().
	const double canny_th2 = 50.0;  // Second threshold for hysteresis procedure in Canny().
	const int canny_aperture_size = 3;  // Aperturesize for the sobel operator in Canny().
	const bool do_merge = false;  // If true, incremental merging of segments will be perfomred.
	cv::Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector(length_threshold, distance_threshold, canny_th1, canny_th2, canny_aperture_size, do_merge);

	// Because of some CPU's power strategy, it seems that the first running of an algorithm takes much longer.
	// So here we run both of the algorithmes 10 times to see each algorithm's processing time with sufficiently warmed-up CPU performance.
	std::vector<cv::Vec4f> lines_lsd, lines_fld;
	for (int run_count = 0; run_count < 10; ++run_count)
	{
		lines_lsd.clear();
		const int64 start_lsd = cv::getTickCount();
		lsd->detect(gray, lines_lsd);
		// Detect the lines with LSD.
		const double freq = cv::getTickFrequency();
		const double duration_ms_lsd = double(cv::getTickCount() - start_lsd) * 1000 / freq;
		std::cout << "Elapsed time for LSD: " << duration_ms_lsd << " ms." << std::endl;

		lines_fld.clear();
		const int64 start_fld = cv::getTickCount();
		// Detect the lines with FLD.
		fld->detect(gray, lines_fld);
		const double duration_ms = double(cv::getTickCount() - start_fld) * 1000 / freq;
		std::cout << "Ealpsed time for FLD " << duration_ms << " ms." << std::endl;
	}

	// Show found lines with LSD.
	cv::Mat line_image_lsd(gray);
	lsd->drawSegments(line_image_lsd, lines_lsd);
	imshow("LSD result", line_image_lsd);

	// Show found lines with FLD.
	cv::Mat line_image_fld(gray);
	fld->drawSegments(line_image_fld, lines_fld);
	cv::imshow("FLD result", line_image_fld);

	cv::waitKey(0);
	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void line()
{
	// Line segment detection (LSD).
	local::lsd();

	// Fast line detector (FLD).
	local::fld();
}

}  // namespace my_opencv
