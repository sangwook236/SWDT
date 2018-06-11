//#include "stdafx.h"
#include "../andres_seeded_region_growing_lib/seeded-region-growing.hxx"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <list>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_region_growing {

// [ref]
//	https://github.com/bjoern-andres/seeded-region-growing
//	http://www.andres.sc/

void andres_seeded_region_growing()
{
	try
	{
#if 0
		const std::string input_filename("./data/segmentation/beach.png");
		const std::string output_filename("./data/segmentation/beach_segmented.png");

		std::list<cv::Point> seed_points;  // (x, y) = (col, row).
		seed_points.push_back(cv::Point(97, 179));
		seed_points.push_back(cv::Point(87, 25));
		seed_points.push_back(cv::Point(120, 84));
		seed_points.push_back(cv::Point(184, 130));
		seed_points.push_back(cv::Point(49, 232));
		const int radius = 2;

		// TODO [adjust] >> values have to be adjusted.
		const int min_intensity = 140, max_intensity = 225;
#elif 0
		const std::string input_filename("./data/segmentation/grain.png");
		const std::string output_filename("./data/segmentation/grain_segmented.png");

		std::list<cv::Point> seed_points;  // (x, y) = (col, row).
		seed_points.push_back(cv::Point(135, 90));
		seed_points.push_back(cv::Point(155, 34));
		seed_points.push_back(cv::Point(83, 140));
		seed_points.push_back(cv::Point(238, 25));
		seed_points.push_back(cv::Point(19, 41));
		seed_points.push_back(cv::Point(14, 166));
		seed_points.push_back(cv::Point(88, 189));
		seed_points.push_back(cv::Point(291, 64));
		const int radius = 2;

		// TODO [adjust] >> values have to be adjusted.
		const int min_intensity = 140, max_intensity = 225;
#elif 1
		const std::string input_filename("./data/segmentation/brain_small.png");
		const std::string output_filename("./data/segmentation/brain_small_segmented.png");

		std::list<cv::Point> seed_points;  // (x, y) = (col, row).
		seed_points.push_back(cv::Point(236, 157));
		seed_points.push_back(cv::Point(284, 310));
		seed_points.push_back(cv::Point(45, 274));
		const int radius = 2;

		// TODO [adjust] >> values have to be adjusted.
		const int min_intensity = 210, max_intensity = 220;
#endif

		const cv::Mat input_img = cv::imread(input_filename, cv::IMREAD_GRAYSCALE);
		if (input_img.empty())
		{
			std::cout << "image file not found: " << input_filename << std::endl;
			return;
		}

        const std::size_t shape[] = { input_img.rows, input_img.cols };
        andres::Marray<unsigned char> elevation(shape, shape + 2);
		andres::Marray<unsigned char> segmentation(shape, shape + 2, 0);

        for (int i = 0; i < input_img.rows; ++i)
			for (int j = 0; j < input_img.cols; ++j)
				elevation(i, j) = input_img.at<unsigned char>(i, j);

		//
#if 0
		{
			const unsigned char maxLevelSeeds = 150;

			boost::timer::auto_cpu_timer timer;
			andres::vision::seededRegionGrowing(elevation, maxLevelSeeds, segmentation);
		}
#elif 0
		{
		    andres::Marray<unsigned char> seeds(elevation.shapeBegin(), elevation.shapeEnd());
			//int idx = 1;
			for (std::list<cv::Point>::const_iterator cit = seed_points.begin(); cit != seed_points.end(); ++cit)
				seeds(cit->y, cit->x) = 1;
				//seeds(cit->y, cit->x) = idx++;  // not correctly working.

			std::vector<size_t> sizes;
			andres::vision::connectedComponentLabeling(seeds, segmentation, sizes);

			boost::timer::auto_cpu_timer timer;
		    andres::vision::seededRegionGrowing(elevation, segmentation);
		}
#elif 1
		{
			cv::Mat seed_img(input_img.size(), input_img.type(), cv::Scalar::all(0));
			for (std::list<cv::Point>::const_iterator cit = seed_points.begin(); cit != seed_points.end(); ++cit)
				cv::circle(seed_img, *cit, radius, CV_RGB(255, 255, 255), CV_FILLED, CV_AA, 0);

			cv::imshow("Andres seeded region growing - seed", seed_img);

		    andres::Marray<unsigned char> seeds(elevation.shapeBegin(), elevation.shapeEnd());
			//int idx = 1;
			for (int i = 0; i < seed_img.rows; ++i)
				for (int j = 0; j < seed_img.cols; ++j)
					if (seed_img.at<unsigned char>(i, j) > 0) seeds(i, j) = 1;
					//if (seed_img.at<unsigned char>(i, j) > 0) seeds(i, j) = idx++;  // not correctly working.

			std::vector<size_t> sizes;
			andres::vision::connectedComponentLabeling(seeds, segmentation, sizes);

			boost::timer::auto_cpu_timer timer;
		    andres::vision::seededRegionGrowing(elevation, segmentation);
		}
#elif 0
		{
			cv::Mat seed_img(input_img.size(), input_img.type(), cv::Scalar::all(0));
			seed_img = min_intensity <= input_img & input_img <= max_intensity;

			cv::imshow("Andres seeded region growing - seed", seed_img);

		    andres::Marray<unsigned char> seeds(elevation.shapeBegin(), elevation.shapeEnd());
			//int idx = 1;
			for (int i = 0; i < seed_img.rows; ++i)
				for (int j = 0; j < seed_img.cols; ++j)
					if (seed_img.at<unsigned char>(i, j) > 0) seeds(i, j) = 1;
					//if (seed_img.at<unsigned char>(i, j) > 0) seeds(i, j) = idx++;  // not correctly working.

			std::vector<size_t> sizes;
			andres::vision::connectedComponentLabeling(seeds, segmentation, sizes);

			boost::timer::auto_cpu_timer timer;
		    andres::vision::seededRegionGrowing(elevation, segmentation);
		}
#endif

		//
		cv::Mat labeling_img(input_img.size(), input_img.type(), cv::Scalar::all(0));
        for (int i = 0; i < input_img.rows; ++i)
			for (int j = 0; j < input_img.cols; ++j)
				labeling_img.at<unsigned char>(i, j) = segmentation(i, j);

        double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(labeling_img, &minVal, &maxVal);
		labeling_img.convertTo(labeling_img, CV_32FC1, 1.0 / maxVal, 0.0);

		cv::imshow("Andres seeded region growing - input", input_img);
		cv::imshow("Andres seeded region growing - label", labeling_img);

		//cv::imwrite(output_filename, labeling_img);

		cv::waitKey(0);

		cv::destroyAllWindows();
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
