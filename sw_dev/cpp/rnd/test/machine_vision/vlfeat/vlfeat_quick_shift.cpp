#include <vl/quickshift.h>
#include <vl/pgm.h>
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <iterator>
#include <algorithm>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_vlfeat {

// [ref] ${VLFEAT_HOME}/toolbox/quickshift/vl_quickshift.c
void quick_shift()
{
	const vl_qs_type kernelSize = 2.0;
	const vl_qs_type maxDist = 20.0;
	const vl_bool medoid = false;  // true to use kernelized medoid shift, false (default) uses quick shift.
	const vl_qs_type ratio = 0.5;  // tradeoff between spatial consistency and color consistency.

	// read image data
	const std::string input_filename = "./data/machine_vision/vlfeat/roofs1.jpg";
	//const std::string input_filename = "./data/machine_vision/opencv/fruits.jpg";

	const cv::Mat input_img = cv::imread(input_filename, cv::IMREAD_COLOR);
	//const cv::Mat input_img = cv::imread(input_filename, cv::IMREAD_GRAYSCALE);
	if (input_img.empty())
	{
		std::cerr << "file not found: " << input_filename << std::endl;
		return;
	}

	double minVal = 0.0, maxVal = 0.0;
	cv::minMaxLoc(input_img, &minVal, &maxVal);

	//
	const int img_width = input_img.cols;
	const int img_height = input_img.rows;
	const int img_numChannels = input_img.channels();

	// channels * width * height + width * row + col
	std::vector<vl_qs_type> image(img_width * img_height * img_numChannels, 0.0);
	{
		if (1 == img_numChannels)
		{
			for (int r = 0; r < img_height; ++r)
				for (int c = 0; c < img_width; ++c)
				{
					//image[img_width * r + c] = input_img.at<unsigned char>(r, c);
					//image[img_width * r + c] = ratio * (input_img.at<unsigned char>(r, c) + cv::randu<vl_qs_type>() / 2550.0);  // not correctly working
					image[img_width * r + c] = ratio * (input_img.at<unsigned char>(r, c) + cv::randu<vl_qs_type>() / 10.0);
				}
		}
		else if (3 == img_numChannels)
		{
			for (int r = 0; r < img_height; ++r)
				for (int c = 0; c < img_width; ++c)
				{
					const cv::Vec3b &pix = input_img.at<cv::Vec3b>(r, c);
					for (int ch = 0; ch < img_numChannels; ++ch)
					{
						//image[ch * img_width * img_height + img_width * r + c] = pix[ch];
						//image[ch * img_width * img_height + img_width * r + c] = ratio * (pix[ch] / 255.0 + cv::randu<vl_qs_type>() / 2550.0);  // not correctly working
						image[ch * img_width * img_height + img_width * r + c] = ratio * (pix[ch] + cv::randu<vl_qs_type>() / 10.0);
					}
				}
		}
		else
		{
			std::cerr << "the number of image's channels is improper ..." << std::endl;
			return;
		}
	}

	//
	VlQS *qs = vl_quickshift_new(&image[0], img_width, img_height, img_numChannels);

	vl_quickshift_set_kernel_size(qs, kernelSize);
	vl_quickshift_set_max_dist(qs, maxDist);
	vl_quickshift_set_medoid(qs, medoid);

	vl_quickshift_process(qs);

	//
	const int *parentsi = vl_quickshift_get_parents(qs);
	const vl_qs_type *dists = vl_quickshift_get_dists(qs);
	const vl_qs_type *density = vl_quickshift_get_density(qs);

	// flatten a tree.
	std::vector<int> parentsv(parentsi, parentsi + img_width * img_height);
	{
		for (int i = 0; i < img_width * img_height; ++i)
			while (parentsv[i] != parentsv[parentsv[i]])
				parentsv[i] = parentsv[parentsv[i]];
	}
/*
	{
		std::vector<int> indexes(parentsv.begin(), parentsv.end());
		std::sort(indexes.begin(), indexes.end());
		std::vector<int>::iterator itEndNew = std::unique(indexes.begin(), indexes.end());
		//const std::size_t labelCount = std::distance(indexes.begin(), itEndNew);
		std::map<int, int> lblMap;
		int idx = 0;
		for (std::vector<int>::iterator it = indexes.begin(); it != itEndNew; ++it, ++idx)
			lblMap[*it] = idx;

		for (std::vector<int>::iterator it = parentsv.begin(); it != parentsv.end(); ++it)
			*it = lblMap[*it];
	}
*/

	// visualize
	cv::Mat result_img(input_img.size(), input_img.type());
	{
		// draw boundary.
		if (1 == result_img.channels())
		{
			vl_size idx = 0;
			for (int r = 0; r < result_img.rows; ++r)
				for (int c = 0; c < result_img.cols; ++c, ++idx)
				{
					const int lbl = parentsv[idx];
					if (r - 1 >= 0 && lbl != parentsv[(r - 1) * result_img.cols + c])
						result_img.at<unsigned char>(r, c) = 255;
					else if (c - 1 >= 0 && lbl != parentsv[r * result_img.cols + (c - 1)])
						result_img.at<unsigned char>(r, c) = 255;
/*
					else if (r + 1 < result_img.rows && lbl != parentsv[(r + 1) * result_img.cols + c])
						result_img.at<unsigned char>(r, c) = 255;
					else if (c + 1 < result_img.cols && lbl != parentsv[r * result_img.cols + (c + 1)])
						result_img.at<unsigned char>(r, c) = 255;
*/
					else
						result_img.at<unsigned char>(r, c) = 0;
				}
		}
		else if (3 == result_img.channels())
		{
			vl_size idx = 0;
			for (int r = 0; r < result_img.rows; ++r)
				for (int c = 0; c < result_img.cols; ++c, ++idx)
				{
					const int lbl = parentsv[idx];
					if (r - 1 >= 0 && lbl != parentsv[(r - 1) * result_img.cols + c])
						result_img.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
					else if (c - 1 >= 0 && lbl != parentsv[r * result_img.cols + (c - 1)])
						result_img.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
/*
					else if (r + 1 < result_img.rows && lbl != parentsv[(r + 1) * result_img.cols + c])
						result_img.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
					else if (c + 1 < result_img.cols && lbl != parentsv[r * result_img.cols + (c + 1)])
						result_img.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
*/
					else
						result_img.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
				}
		}
		else
		{
			std::cerr << "the number of image's channels is improper ..." << std::endl;
			return;
		}
	}

	cv::imshow("quick shift result", result_img);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// clean-up
	vl_quickshift_delete(qs);
}

}  // namespace my_vlfeat
