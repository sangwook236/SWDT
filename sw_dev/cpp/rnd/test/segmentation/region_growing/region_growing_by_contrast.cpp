//#include "stdafx.height"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <boost/math/constants/constants.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>


namespace {
namespace local {

// [ref] ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp
void detect_boundary(const cv::Mat &bk_image, const bool use_8_connectivity, const bool use_zero_padding, cv::Mat &boundary_mask)
{
	cv::Mat input(bk_image.size(), bk_image.type(), cv::Scalar::all(0));
	input.setTo(cv::Scalar::all(1), bk_image > 0);

	// sum filter.
	const int ddepth = -1;  // the output image depth. -1 to use src.depth().
	const int kernel_size = 3;
	const bool normalize = false;
	cv::Mat output;

	const int sum_value = use_8_connectivity ? 9 : 5;
	if (use_8_connectivity)  // use 8-connectivity.
	{
		if (use_zero_padding)
			cv::boxFilter(input, output, ddepth, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1), normalize, cv::BORDER_CONSTANT);
		else
			cv::boxFilter(input, output, ddepth, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1), normalize, cv::BORDER_REPLICATE);
	}
	else  // use 4-connectivity.
	{
		cv::Mat kernel(kernel_size, kernel_size, CV_8UC1, cv::Scalar::all(1));
		kernel.at<unsigned char>(0, 0) = 0; kernel.at<unsigned char>(0, 2) = 0;
		kernel.at<unsigned char>(2, 0) = 0; kernel.at<unsigned char>(2, 2) = 0;

		if (use_zero_padding)
			cv::filter2D(input, output, ddepth, kernel, cv::Point(-1, -1), normalize, cv::BORDER_CONSTANT);
		else
			cv::filter2D(input, output, ddepth, kernel, cv::Point(-1, -1), normalize, cv::BORDER_REPLICATE);
	}

	// 0 if a pixel is in outer region.
	// 1 if a pixel is on outer boundary.
	// 2 if a pixel is on inner boundary.
	// 3 if a pixel is in inner region.

	boundary_mask.setTo(cv::Scalar::all(0));
	boundary_mask.setTo(cv::Scalar::all(1), 0 < output & output < sum_value & 0 == input);
	boundary_mask.setTo(cv::Scalar::all(2), 0 < output & output < sum_value & 1 == input);
	boundary_mask.setTo(cv::Scalar::all(3), sum_value == output & 1 == input);
}

// [ref] imagetool library.
//	${CPP_RND_HOME}/src/image_processing/imagetool.
//	${IMAGETOOL_HOME}/region_growing/regiongrow.h & .c.

void region_growing_by_contrast_in_imagetool(const cv::Mat &input, cv::Mat &region_mask, const double MAX_AVERAGE_CONTRAST, const double MAX_PERIPHERAL_CONTRAST, const int MAX_PIXEL_COUNT_IN_CR, const int xseed, const int yseed)
{
	const int neighbors_x[8] = {  0,  1,  0, -1, -1,  1,  1, -1 };
	const int neighbors_y[8] = { -1,  0,  1,  0, -1, -1,  1,  1 };

	//const int num_neighbors = 4;  // 4-connectivity.
	const int num_neighbors = 8;  // 8-connectivity.

	const int width = input.cols;
	const int height = input.rows;

	std::vector<int> x_sequence(width * height, 0);
	std::vector<int> y_sequence(width * height, 0);
	std::vector<double> average_contrast(width * height, 0.0);
	std::vector<double> peripheral_contrast(width * height, 0.0);

	region_mask.setTo(cv::Scalar::all(0));  // current region (CR) mask.
	region_mask.at<unsigned char>(yseed, xseed) = 255;
	x_sequence[0] = xseed;
	y_sequence[0] = yseed;

	double sum_pixels_in_CR = input.at<unsigned char>(yseed, xseed);  // summation of gray value in current region (CR).
	double sum_pixels_in_IB;  // summation of gray value in internal boundary (IB).
	double sum_pixels_in_CB;  // summation of gray value in current boundary (CB).

	int num_pixels_in_CR = 1;
	int num_pixels_in_IB, num_pixels_in_CB;
	int xmin = xseed, xmax = xseed;
	int ymin = yseed, ymax = yseed;
	int x, y, i, j, k;
	unsigned char pix, max_pix;
	while (true)
	{
		// update the region in consideration.
		xmin = std::min(xmin, std::max(x_sequence[num_pixels_in_CR - 1] - 1, 0));
		xmax = std::max(xmax, std::min(x_sequence[num_pixels_in_CR - 1] + 1, width - 1));
		ymin = std::min(ymin, std::max(y_sequence[num_pixels_in_CR - 1] - 1, 0));
		ymax = std::max(ymax, std::min(y_sequence[num_pixels_in_CR - 1] + 1, height - 1));

		// update the internal boundary and the current boundary.
		sum_pixels_in_IB = sum_pixels_in_CB = 0.0;
		num_pixels_in_IB = num_pixels_in_CB = 0;

		for (y = ymin; y <= ymax; ++y)
		{
			//if (y < 0 || y >= height) continue;

			for (x = xmin; x <= xmax; ++x)
			{
				//if (x < 0 || x >= width) continue;

				pix = input.at<unsigned char>(y, x);
				if (region_mask.at<unsigned char>(y, x) > 0)  // current region.
				{
					for (k = 0; k < num_neighbors; ++k)
					{
						j = x + neighbors_x[k];
						if (j < 0 || j >= width) continue;
						i = y + neighbors_y[k];
						if (i < 0 || i >= height) continue;

						if (0 == region_mask.at<unsigned char>(i, j))  // internal boundary.
						{
							sum_pixels_in_IB += pix;
							++num_pixels_in_IB;
							break;
						}
					}
				}
				else  // not current region.
				{
					for (k = 0; k < num_neighbors; ++k)
					{
						j = x + neighbors_x[k];
						if (j < 0 || j >= width) continue;
						i = y + neighbors_y[k];
						if (i < 0 || i >= height) continue;

						if (region_mask.at<unsigned char>(i, j) > 0)  // current boundary.
						{
							sum_pixels_in_CB += pix;
							++num_pixels_in_CB;
							break;
						}
					}
				}
			}
		}

		average_contrast[num_pixels_in_CR - 1] = std::abs(sum_pixels_in_CR / num_pixels_in_CR - sum_pixels_in_CB / num_pixels_in_CB);
		if (average_contrast[num_pixels_in_CR - 1] > MAX_AVERAGE_CONTRAST) break;

		peripheral_contrast[num_pixels_in_CR - 1] = std::abs(sum_pixels_in_IB / num_pixels_in_IB - sum_pixels_in_CB / num_pixels_in_CB);
		if (peripheral_contrast[num_pixels_in_CR - 1] > MAX_PERIPHERAL_CONTRAST) break;

		if (MAX_PIXEL_COUNT_IN_CR > 0 && num_pixels_in_CR >= MAX_PIXEL_COUNT_IN_CR) break;

		// Find the highest gray pixel in the neighborhood of the current region.
		max_pix = 0;
		for (y = ymin; y <= ymax; ++y)
		{
			//if (y < 0 || y >= height) continue;

			for (x = xmin; x <= xmax; ++x)
			{
				//if (x < 0 || x >= width) continue;

				if (region_mask.at<unsigned char>(y, x) > 0) continue;

				pix = input.at<unsigned char>(y, x);
				if (pix <= max_pix) continue;

				for (k = 0; k < num_neighbors; ++k)
				{
					j = x + neighbors_x[k];
					if (j < 0 || j >= width) continue;
					i = y + neighbors_y[k];
					if (i < 0 || i >= height) continue;

					if (region_mask.at<unsigned char>(i, j) > 0)
					{
						max_pix = pix;
						x_sequence[num_pixels_in_CR] = x;
						y_sequence[num_pixels_in_CR] = y;
						break;
					}
				}
			}
		}

		// if such the pixel exists, mark the pixel. if not, terminate procedure.
		if (0 == max_pix) break;

		region_mask.at<unsigned char>(y_sequence[num_pixels_in_CR], x_sequence[num_pixels_in_CR]) = 255;
		sum_pixels_in_CR += max_pix;

		++num_pixels_in_CR;
	}

#if 0
	std::vector<double>::iterator max_average_contrast_iter = std::max_element(average_contrast.begin(), average_contrast.end());
	std::vector<double>::iterator max_peripheral_contrast_iter = std::max_element(peripheral_contrast.begin(), peripheral_contrast.end());
	std::cout << "max average contrast: " << *max_average_contrast_iter << ", max peripheral contrast: " << *max_peripheral_contrast_iter << std::endl;
	std::cout << "max average contrast idx: " << std::distance(average_contrast.begin(), max_average_contrast_iter) << ", max peripheral contrast idx: " << std::distance(peripheral_contrast.begin(), max_peripheral_contrast_iter) << std::endl;
#endif
}

template<typename T>
void region_growing_by_contrast_using_opencv(const cv::Mat &input, cv::Mat &region_mask, const double MAX_AVERAGE_CONTRAST, const double MAX_PERIPHERAL_CONTRAST, const int MAX_PIXEL_COUNT_IN_CR, const int xseed, const int yseed)
{
	const int neighbors_x[8] = {  0,  1,  0, -1, -1,  1,  1, -1 };
	const int neighbors_y[8] = { -1,  0,  1,  0, -1, -1,  1,  1 };

	//const int num_neighbors = 4;  // 4-connectivity.
	const int num_neighbors = 8;  // 8-connectivity.

	const int width = input.cols;
	const int height = input.rows;

	const bool use_8_connectivity = true;
	const bool use_zero_padding = true;

	std::vector<int> x_sequence, y_sequence;
	x_sequence.reserve(width * height);
	y_sequence.reserve(width * height);
	std::vector<double> average_contrast, peripheral_contrast;
	average_contrast.reserve(width * height);
	peripheral_contrast.reserve(width * height);

	region_mask.setTo(cv::Scalar::all(0));  // current region (CR) mask.
	region_mask.at<unsigned char>(yseed, xseed) = 255;
	x_sequence.push_back(xseed);
	y_sequence.push_back(yseed);

	double sum_pixels_in_CR = input.at<T>(yseed, xseed);  // summation of gray value in current region (CR).
	double sum_pixels_in_IB;  // summation of gray value in internal boundary (IB).
	double sum_pixels_in_CB;  // summation of gray value in current boundary (CB).
	int num_pixels_in_CR = 1;
	int num_pixels_in_IB, num_pixels_in_CB;

	int xmin = xseed, xmax = xseed;
	int ymin = yseed, ymax = yseed;
	T max_pix;
	cv::Mat boundary_mask(input.size(), CV_8UC1), IB_mat(input.size(), input.type()), CB_mat(input.size(), input.type());
	double maxVal;
	cv::Point maxLoc;
	while (true)
	{
		// update the region in consideration.
		xmin = std::min(xmin, std::max(x_sequence.back() - 1, 0));
		xmax = std::max(xmax, std::min(x_sequence.back() + 1, width - 1));
		ymin = std::min(ymin, std::max(y_sequence.back() - 1, 0));
		ymax = std::max(ymax, std::min(y_sequence.back() + 1, height - 1));

#if 0
		detect_boundary(region_mask, use_8_connectivity, use_zero_padding, boundary_mask);
#else
		const cv::Mat &input_roi = input(cv::Range(ymin, ymax + 1), cv::Range(xmin, xmax + 1));
		cv::Mat &region_mask_roi = region_mask(cv::Range(ymin, ymax + 1), cv::Range(xmin, xmax + 1));
		cv::Mat &boundary_mask_roi = boundary_mask(cv::Range(ymin, ymax + 1), cv::Range(xmin, xmax + 1));

		detect_boundary(region_mask_roi, use_8_connectivity, use_zero_padding, boundary_mask_roi);
#endif

		// update the internal boundary and the current boundary.
#if 0
		IB_mat.setTo(cv::Scalar::all(0));
		input.copyTo(IB_mat, 2 == boundary_mask);
		CB_mat.setTo(cv::Scalar::all(0));
		input.copyTo(CB_mat, 1 == boundary_mask);
		sum_pixels_in_IB = cv::sum(IB_mat)[0];
		sum_pixels_in_CB = cv::sum(CB_mat)[0];
		num_pixels_in_IB = cv::countNonZero(IB_mat);
		num_pixels_in_CB = cv::countNonZero(CB_mat);
#else
		cv::Mat &IB_mat_roi = IB_mat(cv::Range(ymin, ymax + 1), cv::Range(xmin, xmax + 1));
		cv::Mat &CB_mat_roi = CB_mat(cv::Range(ymin, ymax + 1), cv::Range(xmin, xmax + 1));
		IB_mat_roi.setTo(cv::Scalar::all(0));
		CB_mat_roi.setTo(cv::Scalar::all(0));

		input_roi.copyTo(IB_mat_roi, 2 == boundary_mask_roi);
		input_roi.copyTo(CB_mat_roi, 1 == boundary_mask_roi);
		sum_pixels_in_IB = cv::sum(IB_mat_roi)[0];
		sum_pixels_in_CB = cv::sum(CB_mat_roi)[0];
		num_pixels_in_IB = cv::countNonZero(IB_mat_roi);
		num_pixels_in_CB = cv::countNonZero(CB_mat_roi);
#endif

		average_contrast.push_back(std::abs(sum_pixels_in_CR / num_pixels_in_CR - sum_pixels_in_CB / num_pixels_in_CB));
		if (average_contrast.back() > MAX_AVERAGE_CONTRAST) break;

		peripheral_contrast.push_back(std::abs(sum_pixels_in_IB / num_pixels_in_IB - sum_pixels_in_CB / num_pixels_in_CB));
		if (peripheral_contrast.back() > MAX_PERIPHERAL_CONTRAST) break;

#if 0
		std::cout << "#pixels in CR: " << num_pixels_in_CR << ", #pixels in IB: " << num_pixels_in_IB << ", #pixels in CB: " << num_pixels_in_CB << std::endl;
		std::cout << "sum in CR: " << sum_pixels_in_CR << ", sum in IB: " << sum_pixels_in_IB << ", sum in CB: " << sum_pixels_in_CB << std::endl;
		std::cout << "average contrast: " << average_contrast.back() << ", peripheral contrast: " << peripheral_contrast.back() << std::endl;
#endif

		if (MAX_PIXEL_COUNT_IN_CR > 0 && num_pixels_in_CR >= MAX_PIXEL_COUNT_IN_CR) break;

		// Find the highest gray pixel in the neighborhood of the current region.
#if 0
		cv::minMaxLoc(CB_mat, NULL, &maxVal, NULL, &maxLoc);
#else
		cv::minMaxLoc(CB_mat_roi, NULL, &maxVal, NULL, &maxLoc);
#endif
		max_pix = maxVal;
		x_sequence.push_back(maxLoc.x);
		y_sequence.push_back(maxLoc.y);

		// if such the pixel exists, mark the pixel. if not, terminate procedure.
		if (0 == max_pix) break;

		region_mask.at<unsigned char>(y_sequence.back(), x_sequence.back()) = 255;
		sum_pixels_in_CR += max_pix;

		++num_pixels_in_CR;
	}

#if 0
	std::vector<double>::iterator max_average_contrast_iter = std::max_element(average_contrast.begin(), average_contrast.end());
	std::vector<double>::iterator max_peripheral_contrast_iter = std::max_element(peripheral_contrast.begin(), peripheral_contrast.end());
	std::cout << "max average contrast: " << *max_average_contrast_iter << ", max peripheral contrast: " << *max_peripheral_contrast_iter << std::endl;
	std::cout << "max average contrast idx: " << std::distance(average_contrast.begin(), max_average_contrast_iter) << ", max peripheral contrast idx: " << std::distance(peripheral_contrast.begin(), max_peripheral_contrast_iter) << std::endl;
#endif
}
	
void region_growing_by_contrast_1()
{
	cv::Mat input_img(400, 400, CV_64FC1, cv::Scalar::all(0));
	if (input_img.empty())
	{
		std::cout << "input image not created" << std::endl;
		return;
	}

	const int ux = 200, uy = 200;
	const int sigma = 25;
	for (int r = 0; r < input_img.rows; ++r)
		for (int c = 0; c < input_img.cols; ++c)
			input_img.at<double>(r, c) = std::exp(-0.5 * (std::pow(double(c - ux) / sigma, 2.0) + std::pow(double(r - uy) / sigma, 2.0)));


	cv::Mat region_mask(input_img.size(), CV_8UC1, cv::Scalar::all(0));
	if (region_mask.empty())
	{
		std::cout << "region mask not created" << std::endl;
		return;
	}

	//
	const double MAX_AVERAGE_CONTRAST = 100.0, MAX_PERIPHERAL_CONTRAST = 100.0;
	const int MAX_PIXEL_COUNT_IN_CR = 25000;
	const int xseed = ux, yseed = uy;

	double minVal, maxVal;
	{
#if 1
		cv::minMaxLoc(input_img, &minVal, &maxVal);
		input_img.convertTo(input_img, CV_8UC1, 255.0 / maxVal, 0.0);

		boost::timer::auto_cpu_timer timer;
		region_growing_by_contrast_in_imagetool(input_img, region_mask, MAX_AVERAGE_CONTRAST, MAX_PERIPHERAL_CONTRAST, MAX_PIXEL_COUNT_IN_CR, xseed, yseed);
		//region_growing_by_contrast_using_opencv<unsigned char>(input_img, region_mask, MAX_AVERAGE_CONTRAST, MAX_PERIPHERAL_CONTRAST, MAX_PIXEL_COUNT_IN_CR, xseed, yseed);
#else
		cv::minMaxLoc(input_img, &minVal, &maxVal);
		input_img.convertTo(input_img, CV_64FC1, 255.0 / maxVal, 0.0);

		boost::timer::auto_cpu_timer timer;
		region_growing_by_contrast_using_opencv<double>(input_img, region_mask, MAX_AVERAGE_CONTRAST, MAX_PERIPHERAL_CONTRAST, MAX_PIXEL_COUNT_IN_CR, xseed, yseed);
#endif
	}

	// display.
	cv::minMaxLoc(input_img, &minVal, &maxVal);
	input_img.convertTo(input_img, CV_32FC1, 1.0 / maxVal, 0.0);

	cv::imshow("region growing by contrast - input", input_img);
	cv::imshow("region growing by contrast - mask", region_mask);

	cv::waitKey(0);

	cv::destroyAllWindows();
}

void region_growing_by_contrast_2()
{
#if 0
	const std::string input_filename("./data/segmentation/beach.png");
	const std::string output_filename("./data/segmentation/beach_segmented.png");

	const int xseed = 97, yseed = 179;
	//const int xseed = 87, yseed = 25;
	//const int xseed = 120, yseed = 84;
	//const int xseed = 184, yseed = 130;
	//const int xseed = 49, yseed = 232;

	// FIXME [adjust] >> adust parameters.
	const double MAX_AVERAGE_CONTRAST = 100.0, MAX_PERIPHERAL_CONTRAST = 100.0;
	const int MAX_PIXEL_COUNT_IN_CR = 10000;
#elif 0
	const std::string input_filename("./data/segmentation/grain.png");
	const std::string output_filename("./data/segmentation/grain_segmented.png");

	const int xseed = 135, yseed = 90;
	//const int xseed = 155, yseed = 34;
	//const int xseed = 83, yseed = 140;
	//const int xseed = 238, yseed = 25;
	//const int xseed = 19, yseed = 41;
	//const int xseed = 14, yseed = 166;
	//const int xseed = 88, yseed = 189;
	//const int xseed = 291, yseed = 64;

	// FIXME [adjust] >> adust parameters.
	const double MAX_AVERAGE_CONTRAST = 100.0, MAX_PERIPHERAL_CONTRAST = 100.0;
	const int MAX_PIXEL_COUNT_IN_CR = 10000;
#elif 1
	const std::string input_filename("./data/segmentation/brain_small.png");
	const std::string output_filename("./data/segmentation/brain_small_segmented.png");

	const int xseed = 236, yseed = 157;
	//const int xseed = 284, yseed = 310;
	//const int xseed = 45, yseed = 274;

	// FIXME [adjust] >> adust parameters.
	const double MAX_AVERAGE_CONTRAST = 100.0, MAX_PERIPHERAL_CONTRAST = 100.0;
	const int MAX_PIXEL_COUNT_IN_CR = 10000;
#endif

	const cv::Mat input_img = cv::imread(input_filename, cv::IMREAD_GRAYSCALE);
	if (input_img.empty())
	{
		std::cout << "image file not found: " << input_filename << std::endl;
		return;
	}

	cv::Mat region_mask(input_img.size(), CV_8UC1, cv::Scalar::all(0));
	if (region_mask.empty())
	{
		std::cout << "region mask not created" << std::endl;
		return;
	}

	{
		boost::timer::auto_cpu_timer timer;
		local::region_growing_by_contrast_in_imagetool(input_img, region_mask, MAX_AVERAGE_CONTRAST, MAX_PERIPHERAL_CONTRAST, MAX_PIXEL_COUNT_IN_CR, xseed, yseed);
		//local::region_growing_by_contrast_using_opencv<unsigned char>(input_img, region_mask, MAX_AVERAGE_CONTRAST, MAX_PERIPHERAL_CONTRAST, MAX_PIXEL_COUNT_IN_CR, xseed, yseed);
	}

	cv::imshow("region growing by contrast - input", input_img);
	cv::imshow("region growing by contrast - mask", region_mask);

	cv::waitKey(0);

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_region_growing {

void region_growing_by_contrast()
{
	try
	{
		//local::region_growing_by_contrast_1();
		local::region_growing_by_contrast_2();
	}
	catch (const cv::Exception &e)
	{
		//std::cout << "OpenCV exception caught: " << e.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception caught:" << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;

		return;
	}
}

}  // namespace my_region_growing
