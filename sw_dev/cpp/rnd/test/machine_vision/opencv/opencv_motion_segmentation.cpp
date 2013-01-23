//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <iterator>
#include <ctime>


namespace {
namespace local {

void make_contour(const cv::Mat &img, const cv::Rect &roi, const int segmentId, std::vector<std::vector<cv::Point> > &contours, std::vector<cv::Vec4i> &hierarchy)
{
	std::vector<std::vector<cv::Point> > contours2;
	if (roi.width == 0 || roi.height == 0)
	{
		cv::findContours((cv::Mat &)img, contours2, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		//cv::findContours((cv::Mat &)img, contours2, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	}
	else
	{
#if defined(__GNUC__)
        {
            cv::Mat img_roi(img, roi);
            cv::findContours(img_roi, contours2, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(roi.x, roi.y));
        }
#else
		cv::findContours(img(roi), contours2, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(roi.x, roi.y));
		//cv::findContours(img : img(roi), contours2, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(roi.x, roi.y));
#endif
	}

	if (contours2.empty()) return;

#if 1
    // comment this out if you do not want approximation
	for (std::vector<std::vector<cv::Point> >::iterator it = contours2.begin(); it != contours2.end(); ++it)
	{
		//if (it->empty()) continue;

		std::vector<cv::Point> approxCurve;
		cv::approxPolyDP(cv::Mat(*it), approxCurve, 3.0, true);
		contours.push_back(approxCurve);
	}
#else
	std::copy(contours2.begin(), contours2.end(), std::back_inserter(contours));
#endif
}

void make_convex_hull(const cv::Mat &img, const cv::Rect &roi, const int segmentId, std::vector<cv::Point> &convexHull)
{
	const cv::Mat &roi_img = roi.width == 0 || roi.height == 0 ? img : img(roi);

	std::vector<cv::Point> points;
	points.reserve(roi_img.rows * roi_img.cols);
	for (int r = 0; r < roi_img.rows; ++r)
		for (int c = 0; c < roi_img.cols; ++c)
		{
			if (roi_img.at<unsigned char>(r, c) == segmentId)
				points.push_back(cv::Point(roi.x + c, roi.y + r));
		}

	if (points.empty()) return;

	std::vector<int> hull;
	cv::convexHull(cv::Mat(points), hull, false);

	if (hull.empty()) return;
	convexHull.reserve(hull.size());

#if 1
	for (std::vector<int>::iterator it = hull.begin(); it != hull.end(); ++it)
		convexHull.push_back(points[*it]);
#else
    // comment this out if you do not want approximation
	std::vector<cv::Point> tmp_points;
	tmp_points.reserve(hull.size());
	for (std::vector<int>::iterator it = hull.begin(); it != hull.end(); ++it)
		tmp_points.push_back(points[*it]);
	cv::approxPolyDP(cv::Mat(tmp_points), convexHull, 3.0, true);
#endif
}

struct IncreaseHierarchyOp
{
    IncreaseHierarchyOp(const int offset)
    : offset_(offset)
    {}

    cv::Vec4i operator()(const cv::Vec4i &rhs) const
    {
        return cv::Vec4i(rhs[0] == -1 ? -1 : (rhs[0] + offset_), rhs[1] == -1 ? -1 : (rhs[1] + offset_), rhs[2] == -1 ? -1 : (rhs[2] + offset_), rhs[3] == -1 ? -1 : (rhs[3] + offset_));
    }

private:
    const int offset_;
};

void segment_motion_using_mhi(const cv::Mat &prev_gray_img, const cv::Mat &curr_gray_img, cv::Mat &mhi, std::vector<std::vector<cv::Point> > &pointSets, std::vector<cv::Vec4i> &hierarchy)
{
	const double timestamp = (double)std::clock() / CLOCKS_PER_SEC;  // get current time in seconds

	const double MHI_DURATION = 1.0;
	const double MAX_TIME_DELTA = 0.5;
	const double MIN_TIME_DELTA = 0.05;

	const int diff_threshold = 8;
	const int motion_gradient_aperture_size = 3;
	const double motion_segment_threshold = MAX_TIME_DELTA;

	pointSets.clear();
	hierarchy.clear();

	cv::Mat silh;
	cv::absdiff(prev_gray_img, curr_gray_img, silh);  // get difference between frames

	cv::threshold(silh, silh, diff_threshold, 1.0, cv::THRESH_BINARY);  // threshold
	cv::updateMotionHistory(silh, mhi, timestamp, MHI_DURATION);  // update MHI

	//
	cv::Mat processed_mhi;  // processed MHI
	{
		const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));
		const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
		const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
		cv::erode(mhi, processed_mhi, selement5);
		cv::dilate(processed_mhi, processed_mhi, selement5);
	}

	// calculate motion gradient orientation and valid orientation mask
	cv::Mat mask;  // valid orientation mask
	cv::Mat orient;  // orientation
	cv::calcMotionGradient(processed_mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, motion_gradient_aperture_size);

	CvMemStorage *storage = cvCreateMemStorage(0);  // temporary storage

	// segment motion: get sequence of motion components
	// segmask is marked motion components map. it is not used further
	IplImage *segmask = cvCreateImage(cvSize(curr_gray_img.cols, curr_gray_img.rows), IPL_DEPTH_32F, 1);  // motion segmentation map
#if defined(__GNUC__)
    IplImage processed_mhi_ipl = (IplImage)processed_mhi;
    CvSeq *seq = cvSegmentMotion(&processed_mhi_ipl, segmask, storage, timestamp, motion_segment_threshold);
#else
	CvSeq *seq = cvSegmentMotion(&(IplImage)processed_mhi, segmask, storage, timestamp, motion_segment_threshold);
#endif

	// FIXME [modify] >>
#if 1
	cv::Mat segmask_id;
	//cv::Mat(segmask, false).convertTo(segmask_id, CV_8SC1, 1.0, 0.0);  // Oops !!! error
	cv::Mat(segmask, false).convertTo(segmask_id, CV_8UC1, 1.0, 0.0);
#elif 0
	// Oops !!! error
	cv::Mat segmask_id(segmask->width, segmask->height, CV_8UC1);
	for (int r = 0; r < segmask_id.rows; ++r)
		for (int c = 0; c < segmask_id.cols; ++c)
			segmask_id.at<unsigned char>(r, c) = (unsigned char)cvRound(CV_IMAGE_ELEM(segmask, float, r, c));
			//segmask_id.at<unsigned char>(r, c) = (unsigned char)cvRound(CV_MAT_ELEM(*segmask, float, r, c));
#else
	// Oops !!! error
	IplImage *segmask_id0 = cvCreateImage(cvSize(segmask->width, segmask->height), IPL_DEPTH_8U, 1);
	cvConvertImage(segmask, segmask_id0, 0);
	const cv::Mat segmask_id(segmask_id0, true);
	cvReleaseImage(&segmask_id0);
#endif

	//
	double minVal = 0.0, maxVal = 0.0;
	//cv::minMaxLoc(segmask, &minVal, &maxVal);
	cv::minMaxLoc(segmask_id, &minVal, &maxVal);

	const size_t max_component_idx((size_t)cvRound(maxVal));
	if (max_component_idx > 0)
	{
		// iterate through the motion components
		pointSets.reserve(seq->total);
		for (int i = 0; i < seq->total; ++i)
		{
			const CvConnectedComp *comp = (CvConnectedComp *)cvGetSeqElem(seq, i);
			const cv::Rect roi(comp->rect);
			if (comp->area < 100 || roi.width + roi.height < 100)  // reject very small components
				continue;

			const size_t count = (size_t)cv::norm(silh(roi), cv::NORM_L1);
			// check for the case of little motion
			if (count < roi.width * roi.height * 0.05)
				continue;

#if 1
			std::vector<std::vector<cv::Point> > contours;
			std::vector<cv::Vec4i> hier;
			make_contour(segmask_id, roi, i, contours, hier);

			if (!hier.empty())
				std::transform(hier.begin(), hier.end(), std::back_inserter(hierarchy), IncreaseHierarchyOp(pointSets.size()));

			for (std::vector<std::vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); ++it)
				if (!it->empty()) pointSets.push_back(*it);
#else
			std::vector<cv::Point> convexHull;
			make_convex_hull(segmask_id, roi, i, convexHull);
			if (!convexHull.empty()) pointSets.push_back(convexHull);
#endif
		}
	}

	cvReleaseImage(&segmask);

	cvClearMemStorage(storage);
	cvReleaseMemStorage(&storage);
}

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
	make_contour(segmask_id, cv::Rect(), 1, contours, hierarchy);

	for (std::vector<std::vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); ++it)
		if (!it->empty()) pointSets.push_back(*it);
#else
	std::vector<cv::Point> convexHull;
	make_convex_hull(segmask_id, cv::Rect(), 1, convexHull);
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

	const bool b1 = capture.set(CV_CAP_PROP_FRAME_WIDTH, imageWidth);
	const bool b2 = capture.set(CV_CAP_PROP_FRAME_HEIGHT, imageHeight);

	const std::string windowName("motion-based segmentation");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	cv::Mat prevgray, gray, frame, frame2;
	cv::Mat mhi, img, blurred;
	std::vector<std::vector<cv::Point> > pointSets;
	std::vector<cv::Vec4i> hierarchy;
	const int maxLevel = 5;
	for (;;)
	{
#if 1
		capture >> frame;
#else
		capture >> frame2;

		if (frame2.cols != imageWidth || frame2.rows != imageHeight)
		{
			//cv::resize(frame2, frame, cv::Size(imageWidth, imageHeight), 0.0, 0.0, cv::INTER_LINEAR);
			cv::pyrDown(frame2, frame);
		}
		else frame = frame2;
#endif

		cv::cvtColor(frame, gray, CV_BGR2GRAY);
		cv::cvtColor(gray, img, CV_GRAY2BGR);

		//if (blurred.empty()) blurred = gray.clone();

		// smoothing
#if 1
		// down-scale and up-scale the image to filter out the noise
		cv::pyrDown(gray, blurred);
		cv::pyrUp(blurred, gray);
#elif 0
		blurred = gray;
		cv::boxFilter(blurred, gray, blurred.type(), cv::Size(5, 5));
#endif

		if (!prevgray.empty())
		{
			if (mhi.empty())
				mhi.create(gray.rows, gray.cols, CV_32F);

			local::segment_motion_using_mhi(prevgray, gray, mhi, pointSets, hierarchy);
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
