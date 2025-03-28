//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>


//#define __USE_ARRAY 1

namespace my_opencv {

void make_convex_hull(const cv::Mat &img, const cv::Rect &roi, const int segmentId, std::vector<cv::Point> &convexHull);
bool simple_convex_hull(const cv::Mat &img, const cv::Rect &roi, const int segmentId, std::vector<cv::Point> &convexHull);

}  // namespace my_opencv

namespace {
namespace local {

void convex_hull_basic()
{
	IplImage *img = cvCreateImage(cvSize(500, 500), 8, 3);

	const char *windowName = "convex hull";
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

#if !defined(__USE_ARRAY)
	CvMemStorage *storage = cvCreateMemStorage();
#endif

	for (;;)
	{
		const int count = rand() % 100 + 1;
		CvPoint pt0;

#if !defined(__USE_ARRAY)
		// generate data points
		CvSeq *ptseq = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage);
		for (int i = 0; i < count; ++i)
		{
			pt0.x = std::rand() % (img->width / 2) + img->width / 4;
			pt0.y = std::rand() % (img->height / 2) + img->height / 4;
			cvSeqPush(ptseq, &pt0);
		}

		// calculate convex hull
		CvSeq *hull = cvConvexHull2(ptseq, NULL, CV_CLOCKWISE, 0);
		const int hullcount = hull->total;
#else
		// generate data points
		CvPoint *points = new CvPoint [count];
		int *hull = new int [count];
		CvMat point_mat = cvMat(1, count, CV_32SC2, points);
		CvMat hull_mat = cvMat(1, count, CV_32SC1, hull);
		for (int i = 0; i < count; ++i)
		{
			pt0.x = std::rand() % (img->width / 2) + img->width / 4;
			pt0.y = std::rand() % (img->height / 2) + img->height / 4;
			points[i] = pt0;
		}

		// calculate convex hull
		cvConvexHull2(&point_mat, &hull_mat, CV_CLOCKWISE, 0);
		const int hullcount = hull_mat.cols;
#endif

		// draw data points
		cvZero(img);
		for (int i = 0; i < count; ++i)
		{
#if !defined(__USE_ARRAY)
			pt0 = *CV_GET_SEQ_ELEM(CvPoint, ptseq, i);
#else
			pt0 = points[i];
#endif
			cvCircle(img, pt0, 2, CV_RGB(255, 0, 0), CV_FILLED);
		}

		// draw convex hull
#if !defined(__USE_ARRAY)
		pt0 = **CV_GET_SEQ_ELEM(CvPoint *, hull, hullcount - 1);
#else
		pt0 = points[hull[hullcount - 1]];
#endif

		for (int i = 0; i < hullcount; ++i)
		{
#if !defined(__USE_ARRAY)
			const CvPoint &pt = **CV_GET_SEQ_ELEM(CvPoint *, hull, i);
#else
			const CvPoint &pt = points[hull[i]];
#endif
			cvLine(img, pt0, pt, CV_RGB(0, 255, 0));
			pt0 = pt;
		}

		cv::imshow(windowName, cv::cvarrToMat(img));

		const int key = cv::waitKey(0);
		if (key == 27)  // ESC
			break;

#if !defined(__USE_ARRAY)
		cvClearMemStorage(storage);
#else
		delete [] points;
		delete [] hull;
#endif
	}

#if !defined(__USE_ARRAY)
	cvReleaseMemStorage(&storage);
#endif

	cvReleaseImage(&img);
	cv::destroyWindow(windowName);
}

void convex_hull_2()
{
	//const std::string input_filename("../data/machine_vision/opencv/thinning_img_1.png");
	const std::string input_filename("../data/machine_vision/opencv/thinning_img_2.jpg");
	cv::Mat src(cv::imread(input_filename));
	if (src.empty())
	{
		std::cerr << "file not found: " << input_filename << std::endl;
		return;
	}

	cv::Mat bw;
	cv::cvtColor(src, bw, cv::COLOR_BGR2GRAY);
	cv::threshold(bw, bw, 10, 255, cv::THRESH_BINARY);

	std::vector<cv::Point> convexHull;
	my_opencv::simple_convex_hull(bw, cv::Rect(), 255, convexHull);

	// display result.
	std::vector<std::vector<cv::Point> > contours;
	contours.push_back(convexHull);
	cv::drawContours(src, contours, 0, CV_RGB(255, 0, 0), 2, 8);

	cv::imshow("convex hull - result", src);

	cv::waitKey();

	cv::destroyAllWindows();
}

void convexity_defect()
{
	IplImage *img = cvCreateImage(cvSize(500, 500), IPL_DEPTH_8U, 3), *gray = cvCreateImage(cvGetSize(img), img->depth, 1);

	const char *windowName = "convexity defect";
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	CvMemStorage *storage = cvCreateMemStorage();

	for (;;)
	{
		const int count = std::rand() % 100 + 5;
		CvPoint pt0;

		// generate data points
		CvSeq *ptseq = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage);
		for (int i = 0; i < count; ++i)
		{
			pt0.x = std::rand() % img->width;
			pt0.y = std::rand() % img->height;
			cvSeqPush(ptseq, &pt0);
		}

		// generate image
		cvZero(img);
		const int npts = 3;
		CvPoint *pts = new CvPoint [npts];
		pts[0] = *CV_GET_SEQ_ELEM(CvPoint, ptseq, 0);
		pts[1] = *CV_GET_SEQ_ELEM(CvPoint, ptseq, 1);
		for (int i = 2; i < count; ++i)
		{
			pts[2] = *CV_GET_SEQ_ELEM(CvPoint, ptseq, i);

			cvFillPoly(img, &pts, &npts, 1, CV_RGB(127, 127, 127), 8, 0);

			pts[0] = pts[1];
			pts[1] = pts[2];
		}
		cvCvtColor(img, gray, cv::COLOR_BGR2GRAY);
		delete [] pts;

		// calculate contours
		CvSeq *contours = NULL;
		cvFindContours(gray, storage, &contours, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

		// comment this out if you do not want approximation
		contours = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, 3, 1);

		// calculate convex hull
		CvSeq *hull = cvConvexHull2(contours, NULL, CV_CLOCKWISE, 0);
		const int hullcount = hull->total;

		// check convexity
		const int isConvex = cvCheckContourConvexity(contours);
		std::cout << (isConvex ? "convex" : "concave") << std::endl;

		// calculate convexity defects
		CvSeq *convexityDefects = cvConvexityDefects(contours, hull, storage);

		// draw data point
		for (int i = 0; i < count; ++i)
		{
			pt0 = *CV_GET_SEQ_ELEM(CvPoint, ptseq, i);
			cvCircle(img, pt0, 1, CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		}

		// draw contours
		const int levels = 5;
		cvDrawContours(img, contours, CV_RGB(0, 0, 255), CV_RGB(0, 127, 127), levels, 1, 8, cvPoint(0, 0));

		// draw convex hull
		pt0 = **CV_GET_SEQ_ELEM(CvPoint *, hull, hullcount - 1);
		for (int i = 0; i < hullcount; ++i)
		{
			const CvPoint &pt = **CV_GET_SEQ_ELEM(CvPoint *, hull, i);
			cvLine(img, pt0, pt, CV_RGB(0, 255, 0), 1, 8, 0);
			pt0 = pt;
		}

		// draw convexity defects
		CvSeq *convexityDefect = convexityDefects;
		while (convexityDefect)
		{
			CvConvexityDefect *defects = new CvConvexityDefect [convexityDefect->total];
			cvCvtSeqToArray(convexityDefect, defects, CV_WHOLE_SEQ);  // copy the contour to a array

			for (int i = 0; i < convexityDefect->total; ++i)
			{
				cvLine(img, *defects[i].start, *defects[i].depth_point, CV_RGB(255, 0, 0), 1, CV_AA, 0);
				cvLine(img, *defects[i].end, *defects[i].depth_point, CV_RGB(255, 0, 0), 1, CV_AA, 0);
				cvCircle(img, *defects[i].depth_point, 2, CV_RGB(255, 0, 0), CV_FILLED, CV_AA, 0);
			}

			delete [] defects;

			// get next contour
			convexityDefect = convexityDefect->h_next;
		}

		cv::imshow(windowName, cv::cvarrToMat(img));

		const int key = cv::waitKey(0);
		if (key == 27)  // ESC
			break;

		cvClearMemStorage(storage);
	}

#if !defined(__USE_ARRAY)
	cvReleaseMemStorage(&storage);
#endif

	cvReleaseImage(&img);
	cv::destroyWindow(windowName);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void convex_hull()
{
	//local::convex_hull_basic();
	local::convex_hull_2();

	//local::convexity_defect();
}

}  // namespace my_opencv
