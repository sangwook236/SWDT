//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <iostream>


namespace {
namespace local {

void chamfer_matching()
{
#if 0
	const std::string filename1("./machine_vision_data/opencv/logo.png");
	const std::string filename2("./machine_vision_data/opencv/logo_in_clutter.png");
#elif 1
	const std::string filename1("./machine_vision_data/opencv/box.png");
	const std::string filename2("./machine_vision_data/opencv/box_in_scene.png");
#elif 0
	const std::string filename1("./machine_vision_data/opencv/melon_target.png");
	//const std::string filename2("./machine_vision_data/opencv/melon_1.png");
	//const std::string filename2("./machine_vision_data/opencv/melon_2.png");
	const std::string filename2("./machine_vision_data/opencv/melon_3.png");
#endif

	cv::Mat templ(cv::imread(filename1, CV_LOAD_IMAGE_GRAYSCALE));
	cv::Mat image(cv::imread(filename2, CV_LOAD_IMAGE_GRAYSCALE));

	// if the image and the template are not edge maps but normal grayscale images, you might want to uncomment the lines below to produce the maps.
	// you can also run Sobel instead of Canny.
	cv::Canny(image, image, 5, 50, 3);
	cv::Canny(templ, templ, 5, 50, 3);

	cv::Mat ctempl;
	cv::cvtColor(templ, ctempl, CV_GRAY2BGR);

	std::vector<std::vector<cv::Point> > results;
	std::vector<float> costs;

	const double templScale = 1.0;
	const int maxMatches = 20;
	const double minMatchDistance = 1.0;
	const int padX = 3, padY = 3;
	const int scales = 5;
	const double minScale = 0.6, maxScale = 1.6;
	const double orientationWeight = 0.5;
	const double truncate = 20;
	const int best_matched_idx = cv::chamerMatching(
		(cv::Mat &)image, (cv::Mat &)templ, results, costs,
		templScale, maxMatches, minMatchDistance, padX, padY,
		scales, minScale, maxScale, orientationWeight, truncate
	);
	if (best_matched_idx < 0)
	{
		std::cout << "object not found" << std::endl;
		return;
	}

	//
	cv::Mat cimg;
	cv::cvtColor(image, cimg, CV_GRAY2BGR);

	const std::vector<cv::Point> &pts = results[best_matched_idx];
	for (std::vector<cv::Point>::const_iterator it = pts.begin(); it != pts.end(); ++it)
	{
		if (it->inside(cv::Rect(0, 0, cimg.cols, cimg.rows)))
			cimg.at<cv::Vec3b>(*it) = cv::Vec3b(0, 255, 0);
	}

	//
	const std::string windowName1("chamfer matching - image");
	const std::string windowName2("chamfer matching - template");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	cv::imshow(windowName1, cimg);
	cv::imshow(windowName2, ctempl);

	cv::waitKey(0);

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void chamfer_matching()
{
	local::chamfer_matching();
}

}  // namespace my_opencv
