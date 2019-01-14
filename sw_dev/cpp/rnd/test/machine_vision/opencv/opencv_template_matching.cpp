//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>


namespace {
namespace local {

void normalized_cross_correlation()
{
	const std::string template_filename("../data/machine_vision/opencv/melon_target.png");
	const std::string img_filename("../data/machine_vision/opencv/melon_1.png");
	//const std::string img_filename("../data/machine_vision/opencv/melon_2.png");
	//const std::string img_filename("../data/machine_vision/opencv/melon_3.png");

	const cv::Mat templ(cv::imread(template_filename, cv::IMREAD_GRAYSCALE));
	if (templ.empty())
	{
		std::cerr << "File not found: " << template_filename << std::endl;
		return;
	}
	cv::Mat img(cv::imread(img_filename, cv::IMREAD_GRAYSCALE));
	if (img.empty())
	{
		std::cerr << "File not found: " << img_filename << std::endl;
		return;
	}

	const int TEMPL_WIDTH = 120;
	const int TEMPL_HEIGHT = 120;

	if (img.rows <= TEMPL_HEIGHT || img.cols <= TEMPL_WIDTH)
	{
		std::cout << "Image is too small." << std::endl;
		return;
	}

	cv::Mat templ_resized;
	cv::resize(templ, templ_resized, cv::Size(TEMPL_WIDTH, TEMPL_HEIGHT), 0.0, 0.0, cv::INTER_LINEAR);

	const size_t pos_x = std::rand() % (img.cols - TEMPL_WIDTH);
	const size_t pos_y = std::rand() % (img.rows - TEMPL_HEIGHT);
	std::cout << "\tInserted location: (" << pos_x << ", " << pos_y << ")" << std::endl;

	// Insert template to image.
#if defined(__GNUC__)
    {
        cv::Mat image_roi(img, cv::Range((int)pos_y, (int)pos_y + TEMPL_HEIGHT), cv::Range((int)pos_x, (int)pos_x + TEMPL_WIDTH));
       	templ_resized.copyTo(image_roi);
    }
#else
	templ_resized.copyTo(img(cv::Range((int)pos_y, (int)pos_y + TEMPL_HEIGHT), cv::Range((int)pos_x, (int)pos_x + TEMPL_WIDTH)));
#endif

	// cv::TM_SQDIFF, cv::TM_SQDIFF_NORMED, cv::TM_CCORR, cv::TM_CCORR_NORMED, cv::TM_CCOEFF, cv::TM_CCOEFF_NORMED.
	const int comparison_method = cv::TM_CCOEFF_NORMED;

	// Perform NCC.
	cv::Mat result;  // A single-channel 32-bit floating-point.
	const double &startTime = (double)cv::getTickCount();
	cv::matchTemplate(img, templ_resized, result, comparison_method);
	const double &endTime = (double)cv::getTickCount();

	double minVal = 0.0, maxVal = 0.0;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

	cv::Point matched_loc(-1, -1);
	switch (comparison_method)
	{
	// Sum of squared differences.
	case cv::TM_SQDIFF:
	case cv::TM_SQDIFF_NORMED:
		std::cout << "\tMin value: " << minVal << std::endl;
		matched_loc = minLoc;
		break;
	// Correlation.
	case cv::TM_CCORR:
	case cv::TM_CCORR_NORMED:
	// Correlation coefficients.
	case cv::TM_CCOEFF:
	case cv::TM_CCOEFF_NORMED:
		std::cout << "\tMax value: " << maxVal << std::endl;
		matched_loc = maxLoc;
		break;
	}

	std::cout << "\tMatched location: (" << matched_loc.x << ", " << matched_loc.y << ")" << std::endl;
	cv::rectangle(img, matched_loc, matched_loc + cv::Point(TEMPL_WIDTH, TEMPL_HEIGHT), CV_RGB(255, 0, 0), 2, cv::LINE_AA, 0);
	std::cout << "\tProcessing time: " << ((endTime - startTime) / ((double)cv::getTickFrequency() * 1000.0)) << " ms" << std::endl;

	//
	const std::string windowName1("Template matching - Image");
	const std::string windowName2("Template matching - Template");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	cv::imshow(windowName1, img);
	cv::imshow(windowName2, templ_resized);

	cv::waitKey(0);

	cv::destroyAllWindows();
}

#define __SCALE_IMAGE_ 1

// REF [site] >> http://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
void simple_multiscale_ncc()
{
	const std::string template_filename("../data/machine_vision/call_of_duty_logo.png");
	std::list<std::string> img_filenames;
	img_filenames.push_back("../data/machine_vision/call_of_duty_1.png");
	img_filenames.push_back("../data/machine_vision/call_of_duty_2.jpg");
	img_filenames.push_back("../data/machine_vision/call_of_duty_3.jpg");
	img_filenames.push_back("../data/machine_vision/call_of_duty_4.jpg");

	// cv::TM_SQDIFF, cv::TM_SQDIFF_NORMED, cv::TM_CCORR, cv::TM_CCORR_NORMED, cv::TM_CCOEFF, cv::TM_CCOEFF_NORMED.
	const int comparison_method = cv::TM_CCOEFF_NORMED;

	const cv::Mat templ(cv::imread(template_filename, cv::IMREAD_GRAYSCALE));
	if (templ.empty())
	{
		std::cerr << "File not found: " << template_filename << std::endl;
		return;
	}

	for (const auto &img_filename : img_filenames)
	{
		const cv::Mat img(cv::imread(img_filename, cv::IMREAD_GRAYSCALE));
		if (img.empty())
		{
			std::cerr << "File not found: " << img_filename << std::endl;
			return;
		}

		//
		cv::Mat templ_edged;
		cv::Canny(templ, templ_edged, 50, 200);
		cv::Mat img_edged;
		cv::Canny(img, img_edged, 50, 200);

		cv::Mat result;  // A single-channel 32-bit floating-point.
		cv::Point matched_loc(-1, -1);
		double matched_val = -std::numeric_limits<double>::max();
		float matched_scale = 0.0f;
		double minVal = 0.0, maxVal = 0.0;
		cv::Point minLoc, maxLoc;
		const double &startTime = (double)cv::getTickCount();

		// NOTICE [caution] >> So sensitive to scale factor.
		//for (float scale = 0.2f; scale < 2.0f; scale *= 1.1f)
		for (float scale = 0.2f; scale < 2.0f; scale += 0.1f)
		{
#if defined(__SCALE_IMAGE_)
			const int width = (int)std::floor(img_edged.cols * scale + 0.5), height = (int)std::floor(img_edged.rows * scale + 0.5);
			if (width < templ_edged.cols || height < templ_edged.rows)
				continue;

#if 1
			cv::Mat img_edged_resized;
			cv::resize(img_edged, img_edged_resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);

			// Perform NCC.
			cv::matchTemplate(img_edged_resized, templ_edged, result, comparison_method);
#else
			cv::Mat img_resized;
			cv::resize(img, img_resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);

			cv::Mat img_resized_edged;
			cv::resize(img_resized, img_resized_edged, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);

			// Perform NCC.
			cv::matchTemplate(img_resized_edged, templ_edged, result, comparison_method);
#endif
#else
			// NOTICE [info] >> Results obtained by scaling a template are bad.

			const int width = (int)std::floor(templ_edged.cols * scale + 0.5), height = (int)std::floor(templ_edged.rows * scale + 0.5);
			if (width > img_edged.cols || height > img_edged.rows)
				continue;

#if 0
			cv::Mat templ_edged_resized;
			cv::resize(templ_edged, templ_edged_resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);

			// Perform NCC.
			cv::matchTemplate(img_edged, templ_edged_resized, result, comparison_method);
#else
			cv::Mat templ_resized;
			cv::resize(templ, templ_resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);

			cv::Mat templ_resized_edged;
			cv::Canny(templ_resized, templ_resized_edged, 50, 200);

			// Perform NCC.
			cv::matchTemplate(img_edged, templ_resized_edged, result, comparison_method);
#endif
#endif
			cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

			switch (comparison_method)
			{
				// Sum of squared differences.
			case cv::TM_SQDIFF:
			case cv::TM_SQDIFF_NORMED:
				maxVal = -minVal;
				maxLoc = minLoc;
				break;
				// Correlation.
			case cv::TM_CCORR:
			case cv::TM_CCORR_NORMED:
				// Correlation coefficients.
			case cv::TM_CCOEFF:
			case cv::TM_CCOEFF_NORMED:
				// Do nothing.
				break;
			}

			if (maxVal > matched_val)
			{
				matched_val = maxVal;
				matched_loc = maxLoc;
				matched_scale = scale;
			}
	}
		const double &endTime = (double)cv::getTickCount();

		cv::Mat rgb;
		cv::cvtColor(img, rgb, cv::COLOR_GRAY2BGR);
#if defined(__SCALE_IMAGE_)
		const int matched_x = (int)std::floor(matched_loc.x / matched_scale + 0.5f), matched_y = (int)std::floor(matched_loc.y / matched_scale + 0.5f);
		const int matched_width = (int)std::floor(templ_edged.cols / matched_scale + 0.5f), matched_height = (int)std::floor(templ_edged.rows / matched_scale + 0.5f);
		std::cout << "\tMatched location = (" << matched_x << ", " << matched_y << "), size = (" << matched_width << ", " << matched_height << "), scale = " << matched_scale << std::endl;
		cv::rectangle(rgb, cv::Point(matched_x, matched_y), cv::Point(matched_x, matched_y) + cv::Point(matched_width, matched_height), CV_RGB(255, 0, 0), 2, cv::LINE_AA, 0);
		std::cout << "\tProcessing time: " << ((endTime - startTime) / ((double)cv::getTickFrequency() * 1000.0)) << " ms" << std::endl;
#else
		const int matched_width = (int)std::floor(templ_edged.cols * matched_scale + 0.5f), matched_height = (int)std::floor(templ_edged.rows * matched_scale + 0.5f);
		std::cout << "\tMatched location = (" << matched_loc.x << ", " << matched_loc.y << "), size = (" << matched_width << ", " << matched_height << "), scale = " << matched_scale << std::endl;
		cv::rectangle(rgb, matched_loc, matched_loc + cv::Point(matched_width, matched_height), CV_RGB(255, 0, 0), 2, cv::LINE_AA, 0);
		std::cout << "\tProcessing time: " << ((endTime - startTime) / ((double)cv::getTickFrequency() * 1000.0)) << " ms" << std::endl;
#endif

		//
		cv::imshow("Template matching - Image", rgb);
		cv::imshow("Template matching - Template", templ);

		cv::waitKey(0);
	}

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void template_matching()
{
	// Normalized cross correlation (NCC).
	//local::normalized_cross_correlation();
	local::simple_multiscale_ncc();
}

}  // namespace my_opencv
