//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <iterator>
#include <ctime>


namespace my_opencv {

void make_contour(const cv::Mat &img, const cv::Rect &roi, const int segmentId, std::vector<std::vector<cv::Point> > &contours, std::vector<cv::Vec4i> &hierarchy);
void make_convex_hull(const cv::Mat &img, const cv::Rect &roi, const int segmentId, std::vector<cv::Point> &convexHull);
void segment_motion_using_mhi(const cv::Mat &prev_gray_img, const cv::Mat &curr_gray_img, cv::Mat &mhi, std::vector<std::vector<cv::Point> > &pointSets, std::vector<cv::Vec4i> &hierarchy);

}  // namespace my_opencv

namespace {
namespace local {

void segment_motion_using_Farneback_motion_estimation(const cv::Mat &prev_gray_img, const cv::Mat &curr_gray_img, std::vector<std::vector<cv::Point> > &pointSets, std::vector<cv::Vec4i> &hierarchy)
{
	pointSets.clear();
	hierarchy.clear();

	const double mag_threshold = 1.0;

	cv::Mat flow;
	cv::calcOpticalFlowFarneback(prev_gray_img, curr_gray_img, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

	cv::Mat segmask_id(flow.rows, flow.cols, CV_8UC1);
	for (int r = 0; r < flow.rows; ++r)
		for (int c = 0; c < flow.cols; ++c)
		{
			const cv::Point2f &fxy = flow.at<cv::Point2f>(r, c);
			segmask_id.at<unsigned char>(r, c) = (fxy.x*fxy.x + fxy.y*fxy.y > mag_threshold ? 1 : 0);
		}

#if 1
	std::vector<std::vector<cv::Point> > contours;
	my_opencv::make_contour(segmask_id, cv::Rect(), 1, contours, hierarchy);

	for (std::vector<std::vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); ++it)
		if (!it->empty()) pointSets.push_back(*it);
#else
	std::vector<cv::Point> convexHull;
	my_opencv::make_convex_hull(segmask_id, cv::Rect(), 1, convexHull);
	if (!convexHull.empty()) pointSets.push_back(convexHull);
#endif
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void motion_segmentation()
{
	const int imageWidth = 640, imageHeight = 480;

	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (!capture.isOpened())
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

	const bool b1 = capture.set(cv::CAP_PROP_FRAME_WIDTH, imageWidth);
	const bool b2 = capture.set(cv::CAP_PROP_FRAME_HEIGHT, imageHeight);

	const std::string windowName("motion-based segmentation");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	cv::Mat prevgray, gray, frame, frame2;
	cv::Mat mhi, img;
	std::vector<std::vector<cv::Point> > pointSets;
	std::vector<cv::Vec4i> hierarchy;
	const int maxLevel = 5;
	for (;;)
	{
#if 1
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}
#else
		capture >> frame2;
		if (frame2.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}

		if (frame2.cols != imageWidth || frame2.rows != imageHeight)
		{
			//cv::resize(frame2, frame, cv::Size(imageWidth, imageHeight), 0.0, 0.0, cv::INTER_LINEAR);
			cv::pyrDown(frame2, frame);
		}
		else frame = frame2;
#endif

		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		cv::cvtColor(gray, img, cv::COLOR_GRAY2BGR);

		// smoothing
#if 1
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
#elif 0
		// METHOD #3: box filtering.

		{
			// FIXME [adjust] >> adjust parameters.
			const int ddepth = -1;  // the output image depth. -1 to use src.depth().
			const int kernelSize = 5;
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

		if (!prevgray.empty())
		{
			if (mhi.empty())
				mhi.create(gray.rows, gray.cols, CV_32F);

			segment_motion_using_mhi(prevgray, gray, mhi, pointSets, hierarchy);
			//local::segment_motion_using_Farneback_motion_estimation(prevgray, gray, pointSets, hierarchy);

			if (!pointSets.empty())
			{
#if 0
				cv::drawContours(img, pointSets, -1, CV_RGB(255, 0, 0), 1, 8, hierarchy, maxLevel, cv::Point());
#elif 0
				const size_t num = pointSets.size();
				for (size_t k = 0; k < num; ++k)
				{
					if (cv::contourArea(cv::Mat(pointSets[k])) < 100.0) continue;
					const int r = rand() % 256, g = rand() % 256, b = rand() % 256;
					cv::drawContours(img, pointSets, k, CV_RGB(r, g, b), 1, 8, hierarchy, maxLevel, cv::Point());
				}
#else
				double maxArea = 0.0;
				size_t maxAreaIdx = -1, idx = 0;
				for (std::vector<std::vector<cv::Point> >::iterator it = pointSets.begin(); it != pointSets.end(); ++it, ++idx)
				{
					const double area = cv::contourArea(cv::Mat(*it));
					if (area > maxArea)
					{
						maxArea = area;
						maxAreaIdx = idx;
					}
				}

				if ((size_t)-1 != maxAreaIdx)
					cv::drawContours(img, pointSets, maxAreaIdx, CV_RGB(255, 0, 0), 1, 8, hierarchy, 0, cv::Point());
					//cv::drawContours(img, pointSets, maxAreaIdx, CV_RGB(255, 0, 0), 1, 8, hierarchy, maxLevel, cv::Point());
#endif
			}
			cv::imshow(windowName, img);
		}

		if (cv::waitKey(1) >= 0)
			break;

		std::swap(prevgray, gray);
	}

	cv::destroyWindow(windowName);
}

}  // namespace my_opencv
