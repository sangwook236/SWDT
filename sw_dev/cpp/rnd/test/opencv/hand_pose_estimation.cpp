//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <iostream>
#include <iterator>
#include <list>
#include <limits>
#include <ctime>


namespace {
namespace local {

void contour(IplImage *srcImg, IplImage *grayImg)
{
	const int levels = 5;
	CvSeq *contours = NULL;
    CvMemStorage *storage = cvCreateMemStorage(0);

    cvFindContours(grayImg, storage, &contours, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

    // comment this out if you do not want approximation
    contours = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, 3, 1);

#if 0
	const int _levels = levels - 3;
	CvSeq *_contours = contours;
    if (_levels <= 0)  // get to the nearest face to make it look more funny
        _contours = _contours->h_next->h_next->h_next;

	cvDrawContours(srcImg, _contours, CV_RGB(255, 0, 0), CV_RGB(0, 255, 0), _levels, 3, CV_AA, cvPoint(0, 0));
#else
	cvDrawContours(srcImg, contours, CV_RGB(255, 0, 0), CV_RGB(0, 255, 0), levels, 2, CV_AA, cvPoint(0, 0));
#endif

    cvReleaseMemStorage(&storage);
}

void snake(IplImage *srcImage, IplImage *grayImage)
{
	const int NUMBER_OF_SNAKE_POINTS = 50;
	const int threshold = 90;

	float alpha = 3;
	float beta = 5;
	float gamma = 2;
	const int use_gradient = 1;
	const CvSize win = cvSize(21, 21);
	const CvTermCriteria term_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, 1.0);

	IplImage *tmp_img = cvCloneImage(grayImage);
	IplImage *img = cvCloneImage(grayImage);

	// make a average filtering
	cvSmooth(tmp_img, img, CV_BLUR, 31, 15);
	//iplBlur(tmp_img, img, 31, 31, 15, 15);  // don't use IPL

	// thresholding
	cvThreshold(img, tmp_img, threshold, 255, CV_THRESH_BINARY);
	//iplThreshold(img, tmp_img, threshold);  // distImg is thresholded image (tmp_img)  // don't use IPL

	// expand the thressholded image of ones -smoothing the edge.
	// and move start position of snake out since there are no ballon force
	cvDilate(tmp_img, img, NULL, 3);

	cvReleaseImage(&tmp_img);

	// find the contours
	CvSeq *contour = NULL;
	CvMemStorage *storage = cvCreateMemStorage(0);
	cvFindContours(img, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	// run through the found coutours
	CvPoint *points = new CvPoint [NUMBER_OF_SNAKE_POINTS];
	while (contour)
	{
		if (contour->total >= NUMBER_OF_SNAKE_POINTS)
		{
			//memset(points, 0, NUMBER_OF_SNAKE_POINTS * sizeof(CvPoint));

			cvSmooth(grayImage, img, CV_BLUR, 7, 3);
			//iplBlur(grayImage, img, 7, 7, 3, 3);  // put blured image in TempImg  // don't use IPL

#if 0
			CvPoint *pts = new CvPoint [contour->total];
			cvCvtSeqToArray(contour, pts, CV_WHOLE_SEQ);  // copy the contour to a array

			// number of jumps between the desired points (downsample only!)
			const int stride = int(contour->total / NUMBER_OF_SNAKE_POINTS);
			for (int i = 0; i < NUMBER_OF_SNAKE_POINTS; ++i)
			{
				points[i].x = pts[int(i * stride)].x;
				points[i].y = pts[int(i * stride)].y;
			}

			delete [] pts;
			pts = NULL;
#else
			const int stride = int(contour->total / NUMBER_OF_SNAKE_POINTS);
			for (int i = 0; i < NUMBER_OF_SNAKE_POINTS; ++i)
			{
				CvPoint *pt = CV_GET_SEQ_ELEM(CvPoint, contour, i * stride);
				points[i].x = pt->x;
				points[i].y = pt->y;
			}
#endif

			// snake
			cvSnakeImage(img, points, NUMBER_OF_SNAKE_POINTS, &alpha, &beta, &gamma, CV_VALUE, win, term_criteria, use_gradient);

			// draw snake on image
			cvPolyLine(srcImage, (CvPoint **)&points, &NUMBER_OF_SNAKE_POINTS, 1, 1, CV_RGB(255, 0, 0), 3, 8, 0);
		}

		// get next contours
		contour = contour->h_next;
	}

	//
	//free(contour);
	delete [] points;

	cvReleaseMemStorage(&storage);
	cvReleaseImage(&img);
}

void mser(IplImage *srcImage, IplImage *grayImage)
{
	IplImage *hsv = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_8U, 3);
	cvCvtColor(srcImage, hsv, CV_BGR2YCrCb);

	CvMSERParams params = cvMSERParams();  //cvMSERParams(5, 60, cvRound(0.2 * grayImage->width * grayImage->height), 0.25, 0.2);
	CvMemStorage *storage= cvCreateMemStorage();
	CvSeq *contours = NULL;
	double t = (double)cvGetTickCount();
	cvExtractMSER(hsv, NULL, &contours, storage, params);
	t = cvGetTickCount() - t;

	cvReleaseImage(&hsv);

	std::cout << "MSER extracted " << contours->total << " contours in " << (t/((double)cvGetTickFrequency()*1000.0)) << " ms" << std::endl;

	// draw MSER with different color
	//unsigned char *imgptr = (unsigned char *)srcImage->imageData;
	//for (int i = contours->total - 1; i >= 0; --i)
	//{
	//	CvSeq *seq = *(CvSeq **)cvGetSeqElem(contours, i);
	//	for (int j = 0; j < seq->total; ++j)
	//	{
	//		CvPoint *pt = CV_GET_SEQ_ELEM(CvPoint, seq, j);
	//		imgptr[pt->x*3+pt->y*srcImage->widthStep] = bcolors[i%9][2];
	//		imgptr[pt->x*3+1+pt->y*srcImage->widthStep] = bcolors[i%9][1];
	//		imgptr[pt->x*3+2+pt->y*srcImage->widthStep] = bcolors[i%9][0];
	//	}
	//}

	// find ellipse ( it seems cvFitEllipse2 have error or sth? )
	// FIXME [check] >> there are some errors. have to compare original source (mser_sample.cpp)
	for (int i = 0; i < contours->total; ++i)
	{
		const CvContour *contour = *(CvContour **)cvGetSeqElem(contours, i);
		const CvBox2D box = cvFitEllipse2(contour);
		//box.angle = (float)CV_PI / 2.0f - box.angle;

		cvEllipseBox(srcImage, box, CV_RGB(255, 0, 0), 2, 8, 0);
	}

	cvClearMemStorage(storage);
}

void make_contour(const cv::Mat &segmentMask, const cv::Rect &roi, const int segmentId, std::vector<std::vector<cv::Point> > &contours, std::vector<cv::Vec4i> &hierarchy)
{
	std::vector<std::vector<cv::Point> > contours2;
#if defined(__GNUC__)
    {
        cv::Mat segmentMask_tmp(roi.width == 0 || roi.height == 0 ? segmentMask : segmentMask(roi));
        cv::findContours(segmentMask_tmp, contours2, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(roi.x, roi.y));
    }
#else
	cv::findContours(roi.width == 0 || roi.height == 0 ? segmentMask : segmentMask(roi), contours2, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(roi.x, roi.y));
#endif
	if (roi.width == 0 || roi.height == 0)
	{
		cv::findContours((cv::Mat &)segmentMask, contours2, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		//cv::findContours((cv::Mat &)segmentMask, contours2, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	}
	else
	{
#if defined(__GNUC__)
        {
            cv::Mat segmentMask_roi(segmentMask(roi));
            cv::findContours(segmentMask_roi, contours2, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(roi.x, roi.y));
            //cv::findContours(segmentMask_roi, contours2, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(roi.x, roi.y));
        }
#else
		cv::findContours(segmentMask(roi), contours2, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(roi.x, roi.y));
		//cv::findContours(segmentMask(roi), contours2, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(roi.x, roi.y));
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

void make_convex_hull(const cv::Mat &segmentMask, const cv::Rect &roi, const int segmentId, std::vector<cv::Point> &convexHull)
{
	const cv::Mat &roi_img = roi.width == 0 || roi.height == 0 ? segmentMask : segmentMask(roi);

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

void segment_motion_using_mhi(const bool useConvexHull, const cv::Mat &prev_gray_img, const cv::Mat &curr_gray_img, cv::Mat &mhi, cv::Mat &segmentMask, std::vector<std::vector<cv::Point> > &pointSets, std::vector<cv::Vec4i> &hierarchy)
{
	const double timestamp = (double)clock() / CLOCKS_PER_SEC;  // get current time in seconds

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

	cv::threshold(silh, silh, diff_threshold, 1.0, cv::THRESH_BINARY);  // threshold it
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

	//
	//cv::Mat(segmask, false).convertTo(segmentMask, CV_8SC1, 1.0, 0.0);  // Oops !!! error
	cv::Mat(segmask, false).convertTo(segmentMask, CV_8UC1, 1.0, 0.0);

	//
	double minVal = 0.0, maxVal = 0.0;
	//cv::minMaxLoc(segmask, &minVal, &maxVal);
	cv::minMaxLoc(segmentMask, &minVal, &maxVal);
	if (maxVal < 1.0e-5) return;

	// iterate through the motion components
	pointSets.reserve(seq->total);
	for (int i = 0; i < seq->total; ++i)
	{
		const CvConnectedComp *comp = (CvConnectedComp *)cvGetSeqElem(seq, i);
		const cv::Rect roi(comp->rect);
		if (comp->area < 100 || roi.width + roi.height < 100)  // reject very small components
			continue;

		if (useConvexHull)
		{
			std::vector<cv::Point> convexHull;
			make_convex_hull(segmentMask, roi, i, convexHull);
			if (!convexHull.empty()) pointSets.push_back(convexHull);
		}
		else
		{
			std::vector<std::vector<cv::Point> > contours;
			std::vector<cv::Vec4i> hier;
			make_contour(segmentMask, roi, i, contours, hier);

			if (!hier.empty())
				std::transform(hier.begin(), hier.end(), std::back_inserter(hierarchy), IncreaseHierarchyOp(pointSets.size()));
			std::copy(contours.begin(), contours.end(), std::back_inserter(pointSets));
		}
	}

	cvReleaseImage(&segmask);

	cvClearMemStorage(storage);
	cvReleaseMemStorage(&storage);
}

void segment_motion_using_Farneback_motion_estimation(const bool useConvexHull, const cv::Mat &prev_gray_img, const cv::Mat &curr_gray_img, cv::Mat &segmentMask, std::vector<std::vector<cv::Point> > &pointSets, std::vector<cv::Vec4i> &hierarchy)
{
	pointSets.clear();
	hierarchy.clear();

	cv::Mat flow;
	cv::calcOpticalFlowFarneback(prev_gray_img, curr_gray_img, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

	segmentMask = cv::Mat::zeros(flow.rows, flow.cols, CV_8UC1);
#if 0
	const double mag_threshold = 1.0;
	for (int r = 0; r < flow.rows; ++r)
		for (int c = 0; c < flow.cols; ++c)
		{
			const cv::Point2f &fxy = flow.at<cv::Point2f>(r, c);
			segmentMask.at<unsigned char>(r, c) = (fxy.x*fxy.x + fxy.y*fxy.y > mag_threshold ? 1 : 0);
		}
#else
	std::multimap<double, cv::Point2i> mag_pos_pairs;
	for (int r = 0; r < flow.rows; ++r)
		for (int c = 0; c < flow.cols; ++c)
		{
			const cv::Point2f &fxy = flow.at<cv::Point2f>(r, c);
			mag_pos_pairs.insert(std::make_pair(fxy.x*fxy.x + fxy.y*fxy.y, cv::Point2i(c, r)));
		}

	const double threshold_ratio = 0.2;
	const size_t numPairs = mag_pos_pairs.size();
	const size_t lower_bound_num = (size_t)std::floor(numPairs * threshold_ratio);
	std::multimap<double, cv::Point2i>::iterator itBegin = mag_pos_pairs.begin();
	std::advance(itBegin, lower_bound_num);
	for (std::multimap<double, cv::Point2i>::iterator it = itBegin; it != mag_pos_pairs.end(); ++it)
		segmentMask.at<unsigned char>(it->second.y, it->second.x) = 1;
#endif

	if (useConvexHull)
	{
		std::vector<cv::Point> convexHull;
		make_convex_hull(segmentMask, cv::Rect(), 1, convexHull);
		if (!convexHull.empty()) pointSets.push_back(convexHull);
	}
	else
	{
		make_contour(segmentMask, cv::Rect(), 1, pointSets, hierarchy);
	}
}

void fit_contour_by_snake(const cv::Mat &gray_img, const std::vector<cv::Point> &contour, const size_t numSnakePoints, std::vector<cv::Point> &snake_contour)
{
	snake_contour.clear();
	if (contour.empty()) return;

/*
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Point> hierarchy;
	{
		const int threshold = 90;

		cv::Mat binary_img;

		// make a average filtering
		cv::blur(gray_img, binary_img, cv::Size(31, 15));

		// thresholding
		cv::threshold(binary_img, binary_img, threshold, 255, cv::THRESH_BINARY);

		// expand the thressholded image of ones -smoothing the edge.
		// and move start position of snake out since there are no ballon force
		{
			const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
			cv::dilate(binary_img, binary_img, selement, cv::Point(-1, -1), 3);
		}

		cv::findContours(binary_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	}
*/

	float alpha = 3;
	float beta = 5;
	float gamma = 2;
	const int use_gradient = 1;
	const CvSize win = cvSize(21, 21);
	const CvTermCriteria term_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, 1.0);

	// run through the found coutours
	const size_t &numPts = contour.size();
	const size_t numSnakePts = 0 == numSnakePoints ? numPts : numSnakePoints;
	if (numPts >= numSnakePts)
	{
		CvPoint *points = new CvPoint [numSnakePts];

		cv::Mat blurred_img;
		cv::blur(gray_img, blurred_img, cv::Size(7, 3));

		const int stride = int(numPts / numSnakePts);
		for (size_t i = 0; i < numSnakePts; ++i)
		{
			const cv::Point &pt = contour[i * stride];
			points[i] = cvPoint(pt.x, pt.y);
		}

		// snake
#if defined(__GNUC__)
        IplImage blurred_img_ipl = (IplImage)blurred_img;
		cvSnakeImage(&blurred_img_ipl, points, numSnakePts, &alpha, &beta, &gamma, CV_VALUE, win, term_criteria, use_gradient);
#else
		cvSnakeImage(&(IplImage)blurred_img, points, numSnakePts, &alpha, &beta, &gamma, CV_VALUE, win, term_criteria, use_gradient);
#endif

		snake_contour.assign(points, points + numSnakePts);
		delete [] points;
	}
}

void calculateDistanceTransform(const cv::Mat &gray, cv::Mat &distanceTransform)
{
	const int distanceType = CV_DIST_C;  // C/Inf metric
	//const int distanceType = CV_DIST_L1;  // L1 metric
	//const int distanceType = CV_DIST_L2;  // L2 metric
	//const int maskSize = CV_DIST_MASK_3;
	//const int maskSize = CV_DIST_MASK_5;
	const int maskSize = CV_DIST_MASK_PRECISE;

#if 0
	cv::Mat dist32f1, dist32f2;
	// distance transform of original image
	cv::distanceTransform(gray, dist32f1, distanceType, maskSize);
	// distance transform of inverted image
	cv::distanceTransform(cv::Scalar::all(255) - gray, dist32f2, distanceType, maskSize);
	cv::max(dist32f1, dist32f2, distanceTransform);
#else
	// distance transform of inverted image
	cv::distanceTransform(gray, distanceTransform, distanceType, maskSize);
#endif
}

void findHandContour(const cv::Mat &edgeImg, std::vector<std::vector<cv::Point> > &contours)
{
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours((cv::Mat &)edgeImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	//cv::findContours((cv::Mat &)edgeImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}

void findBetterHandContour(const cv::Size &imgSize, const std::vector<std::vector<cv::Point> > &contours, const std::vector<cv::Point> &convexHull, const size_t contourId, const cv::Point &seed, std::vector<std::vector<cv::Point> > &silhouetteContours)
{
	cv::Mat gray(imgSize, CV_8UC1, cv::Scalar::all(0));

	{
		const cv::Point *h = (const cv::Point *)&convexHull[0];
		const size_t num = convexHull.size();
		cv::polylines(gray, (const cv::Point **)&h, (int *)&num, 1, true, CV_RGB(128, 128, 128), 1, 8, 0);
		cv::drawContours(gray, contours, contourId, CV_RGB(255, 255, 255), 1, 8);
		cv::floodFill(gray, seed, CV_RGB(255, 255, 255));
	}

	gray = gray > 192;

	//const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	//const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));
	const int iterations = 1;
	cv::morphologyEx(gray, gray, cv::MORPH_OPEN, selement, cv::Point(-1, -1), iterations);

	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(gray, silhouetteContours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}

void findBetterHandContour(const cv::Size &imgSize, const std::vector<cv::Point> &contour, const std::vector<cv::Point> &convexHull, const cv::Point &seed, std::vector<std::vector<cv::Point> > &silhouetteContours)
{
	cv::Mat gray(imgSize, CV_8UC1, cv::Scalar::all(0));

	{
		const cv::Point *h1 = (const cv::Point *)&convexHull[0];
		const size_t num1 = convexHull.size();
		cv::polylines(gray, (const cv::Point **)&h1, (int *)&num1, 1, true, CV_RGB(128, 128, 128), 1, 8, 0);
		const cv::Point *h2 = (const cv::Point *)&contour[0];
		const size_t num2 = contour.size();
		cv::polylines(gray, (const cv::Point **)&h2, (int *)&num2, 1, true, CV_RGB(255, 255, 255), 1, 8, 0);
		cv::floodFill(gray, seed, CV_RGB(255, 255, 255));
	}

	gray = gray > 192;

	//const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	//const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));
	const int iterations = 1;
	cv::morphologyEx(gray, gray, cv::MORPH_OPEN, selement, cv::Point(-1, -1), iterations);

	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(gray, silhouetteContours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}

void findHandSilhouette(const cv::Size &imgSize, const std::vector<std::vector<cv::Point> > &contours, const std::vector<cv::Point> &convexHull, const size_t contourId, const cv::Scalar &hullMeanPt, std::vector<std::vector<cv::Point> > &silhouetteContours, cv::Point &palmCenterPoint)
{
	if (contours.empty() || contours[contourId].empty() || convexHull.empty()) return;

	cv::Mat gray(imgSize, CV_8UC1, cv::Scalar::all(255));
	cv::drawContours(gray, contours, contourId, CV_RGB(0, 0, 0), 1, 8);

	cv::Mat distanceTransform;  // CV_32FC1
	calculateDistanceTransform(gray, distanceTransform);

	cv::Mat mask(imgSize, CV_8UC1, cv::Scalar::all(0));
	{
		const cv::Point *h = (const cv::Point *)&convexHull[0];
		const size_t num = convexHull.size();
		cv::polylines(mask, (const cv::Point **)&h, (int *)&num, 1, true, CV_RGB(255, 255, 255), 1, 8, 0);
		cv::floodFill(mask, cv::Point(cvRound(hullMeanPt[0]), cvRound(hullMeanPt[1])), CV_RGB(255, 255, 255));
	}

	// FIXME [check] >>
	// palmCenterPoint has a strange value, but the result is correct ???
	cv::minMaxLoc(distanceTransform, NULL, NULL, NULL, &palmCenterPoint, mask);

	//
	findBetterHandContour(imgSize, contours, convexHull, contourId, palmCenterPoint, silhouetteContours);
}

void findConvexityDefect(CvMemStorage *storage, const std::vector<cv::Point> &contour, const std::vector<cv::Point> &convexHull, const double distanceThreshold, const double depthThreshold, std::vector<std::vector<CvConvexityDefect> > &convexityDefects, std::vector<cv::Point> &convexityDefectPoints)
{
	CvSeq *contourSeq = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage);
	CvPoint pt;
	for (std::vector<cv::Point>::const_iterator it = contour.begin(); it != contour.end(); ++it)
	{
		pt.x = it->x;
		pt.y = it->y;
		cvSeqPush(contourSeq, &pt);
	}

	// FIXME [modify] >> Oops !!! too stupid
	// convex hull is already calculated by cv::convexHull()
	CvSeq *hullSeq = cvConvexHull2(contourSeq, NULL, CV_CLOCKWISE, 0);

	CvSeq *convexityDefectSeq = cvConvexityDefects(contourSeq, hullSeq, storage);
	while (convexityDefectSeq)
	{
		CvConvexityDefect *defects = new CvConvexityDefect [convexityDefectSeq->total];
		cvCvtSeqToArray(convexityDefectSeq, defects, CV_WHOLE_SEQ);  // copy the contour to a array

#if 0
		convexityDefects.push_back(std::vector<CvConvexityDefect>(defects, defects + convexityDefectSeq->total));
#else
		// eliminate interior convexity defect
		// FIXME [modify] >> Oops !!! stupid implementation
		std::vector<CvConvexityDefect> dfts;
		dfts.reserve(convexityDefectSeq->total);
		for (int i = 0; i < convexityDefectSeq->total; ++i)
		{
			bool pass = true;

#if 0
			cv::Point pt1, pt2;

			std::vector<cv::Point>::const_iterator it = contour.begin();
			pt1 = *it;
			++it;
			for (; it != contour.end(); ++it)
			{
				pt2 = *it;

				if ((cv::norm(pt1 - cv::Point(defects[i].start->x, defects[i].start->y)) <= distanceThreshold &&
					cv::norm(pt2 - cv::Point(defects[i].end->x, defects[i].end->y)) <= distanceThreshold) ||
					(cv::norm(pt1 - cv::Point(defects[i].end->x, defects[i].end->y)) <= distanceThreshold &&
					cv::norm(pt2 - cv::Point(defects[i].start->x, defects[i].start->y)) <= distanceThreshold) ||
					defects[i].depth <= depthThreshold)
				{
					pass = false;
					break;
				}

				pt1 = pt2;
			}
#else
			// FIXME [check] >> is it really correct?
			cv::Point pt1, pt2;

			std::vector<cv::Point>::const_iterator it = convexHull.begin();
			pt1 = *it;
			++it;
			for (; it != convexHull.end(); ++it)
			{
				pt2 = *it;

				const double a = pt2.x == pt1.x ? 0 : (pt2.y - pt1.y) / (pt2.x - pt1.x);
				const double b = -a * pt1.x + pt1.y;
				const double d = std::fabs(a * defects[i].depth_point->x - defects[i].depth_point->y + b) / std::sqrt(a*a + 1);
				if (d <= distanceThreshold || defects[i].depth <= depthThreshold)
				{
					pass = false;
					break;
				}

				pt1 = pt2;
			}
#endif

			if (pass) dfts.push_back(defects[i]);
		}
		convexityDefects.push_back(dfts);
#endif

		delete [] defects;

		// get next contour
		convexityDefectSeq = convexityDefectSeq->h_next;
	}

	// calculate a point on palm
	std::vector<cv::Point> defectPts;
	for (std::vector<std::vector<CvConvexityDefect> >::const_iterator it = convexityDefects.begin(); it != convexityDefects.end(); ++it)
		for (std::vector<CvConvexityDefect>::const_iterator itDefect = it->begin(); itDefect != it->end(); ++itDefect)
			defectPts.push_back(cv::Point(itDefect->depth_point->x, itDefect->depth_point->y));

	// FIXME [restore] >>
	// sort convexity defect contour
	//cv::convexHull(cv::Mat(defectPts), convexityDefectPoints, false);
	convexityDefectPoints.swap(defectPts);
}

bool segmentFinger(cv::Mat &img, const cv::Point &startPt, const cv::Point &endPt, const std::vector<cv::Point> &handContour, const double distanceThreshold, std::vector<std::vector<cv::Point> > &fingerContours)
{
	// FIXME [enhance] >> searching speed

	int flag = 0;
	bool is_final = false;

	std::vector<cv::Point> fingerContour;
	for (std::vector<cv::Point>::const_iterator itPt = handContour.begin(); itPt != handContour.end(); ++itPt)
	{
		if (cv::norm(*itPt - startPt) <= distanceThreshold)
		{
			if (flag)
			{
				std::cout << "start point error !!!" << std::endl;
				return false;
			}
			else flag = -1;
		}
		else if (cv::norm(*itPt - endPt) <= distanceThreshold)
		{
			if (flag)
			{
				if (1 == flag)
				{
					std::cout << "end point error !!!" << std::endl;
					return false;
				}
				else
				{
					fingerContour.push_back(*itPt);
					break;
				}
			}
			else
			{
				is_final = true;
			}
		}

		if (!flag) continue;

		fingerContour.push_back(*itPt);
	}

	if (is_final)
	{
		for (std::vector<cv::Point>::const_iterator itPt = handContour.begin(); itPt != handContour.end(); ++itPt)
		{
			if (cv::norm(*itPt - endPt) <= distanceThreshold)
			{
				if (flag)
				{
					if (1 == flag)
					{
						std::cout << "end point error !!!" << std::endl;
						return false;
					}
					else
					{
						fingerContour.push_back(*itPt);
						break;
					}
				}
				else break;
			}

			if (!flag) continue;

			fingerContour.push_back(*itPt);
		}
	}

	fingerContours.push_back(fingerContour);
	return true;
}

bool calculateCurvature(const cv::Point2d &v1, const cv::Point2d &v2, double &curvature)
{
	const double norm1(cv::norm(v1)), norm2(cv::norm(v2));
	const double eps = 1.0e-15;

	if (norm1 <= eps || norm2 <= eps) return false;
	else
	{
		// 0.0 <= curvature <= 1.0. when 1.0, max curvature(the same direction). when 0.0, min curvature(opposite direction)
		curvature = 0.5 * (1.0 + (v1.x*v2.x + v1.y*v2.y) / (norm1 * norm2));
		//curvature = 0.5 * (1.0 + cv::Mat(v1).dot(cv::Mat(v2)) / (norm1 * norm2));

		// CW or CCW
		if (v2.x*v1.y - v1.x*v2.y < 0.0)  // v2 x v1
			curvature = -curvature;

		return true;
	}
}

bool findCurvaturePoints(const std::vector<cv::Point> &fingerContour, const size_t displaceIndex, size_t &minIdx, size_t &maxIdx)
{
	minIdx = maxIdx = -1;

	const size_t &num = fingerContour.size();
	if (num < 2 * displaceIndex + 1) return false;

	const size_t endIdx = num - 2 * displaceIndex;

	double minCurvature = std::numeric_limits<double>::max();
	double maxCurvature = -std::numeric_limits<double>::max();
	double curvature;
	for (size_t i = 0; i < endIdx; ++i)
	{
	    const cv::Point &pt1 = fingerContour[i] - fingerContour[displaceIndex + i];
	    const cv::Point &pt2 = fingerContour[2 * displaceIndex + i] - fingerContour[displaceIndex + i];
		if (calculateCurvature(cv::Point2d(pt1.x, pt1.y), cv::Point2d(pt2.x, pt2.y), curvature))
		{
			if (curvature < minCurvature)
			{
				minCurvature = curvature;
				minIdx = displaceIndex + i;
			}
			if (curvature > maxCurvature)
			{
				maxCurvature = curvature;
				maxIdx = displaceIndex + i;
			}
		}
	}

	return true;
}

void findFingertips(const std::vector<std::vector<cv::Point> > &fingerContours, const size_t displaceIndex, const cv::Point &palmCenterPoint, const cv::Point &hullCenterPoint, std::vector<cv::Point> &fingertips)
{
#if 1
	const size_t &count = fingerContours.size();
	std::vector<cv::Point> minFingerTipPoints, maxFingerTipPoints;
	minFingerTipPoints.reserve(count);
	maxFingerTipPoints.reserve(count);
	for (std::vector<std::vector<cv::Point> >::const_iterator it = fingerContours.begin(); it != fingerContours.end(); ++it)
	{
		size_t minIdx = -1, maxIdx = -1;
		if (findCurvaturePoints(*it, displaceIndex, minIdx, maxIdx))
		{
			if (-1 != minIdx) minFingerTipPoints.push_back((*it)[minIdx]);
			if (-1 != maxIdx) maxFingerTipPoints.push_back((*it)[maxIdx]);
		}
	}

	fingertips.swap(minFingerTipPoints);
#elif 0
	const size_t &count = fingerContours.size();
	fingertips.reserve(count);
	for (std::vector<std::vector<cv::Point> >::const_iterator it = fingerContours.begin(); it != fingerContours.end(); ++it)
	{
		size_t minIdx = -1, maxIdx = -1;
		if (findCurvaturePoints(*it, displaceIndex, minIdx, maxIdx))
		{
			if (-1 != minIdx && -1 != maxIdx)
			{
				const cv::Point &pt1 = (*it)[minIdx], &pt2 = (*it)[maxIdx];
				const double len1 = (pt1.x - palmCenterPoint.x) * (pt1.x - palmCenterPoint.x) + (pt1.y - palmCenterPoint.y) * (pt1.y - palmCenterPoint.y);
				const double len2 = (pt2.x - palmCenterPoint.x) * (pt2.x - palmCenterPoint.x) + (pt2.y - palmCenterPoint.y) * (pt2.y - palmCenterPoint.y);
				fingertips.push_back((*it)[len1 >= len2 ? minIdx : maxIdx]);
			}
			else if (-1 != minIdx) fingertips.push_back((*it)[minIdx]);
			else if (-1 != maxIdx) fingertips.push_back((*it)[maxIdx]);
		}
	}
#else
	cv::Point maxPt;
	for (std::vector<std::vector<cv::Point> >::const_iterator it = fingerContours.begin(); it != fingerContours.end(); ++it)
	{
		double maxDist = 0.0;
		bool flag = false;
		for (std::vector<cv::Point>::const_iterator itPoint = it->begin(); itPoint != it->end(); ++itPoint)
		{
			const double dist = (itPoint->x - palmCenterPoint.x) * (itPoint->x - palmCenterPoint.x) + (itPoint->y - palmCenterPoint.y) * (itPoint->y - palmCenterPoint.y);
			if (dist > maxDist)
			{
				maxDist = dist;
				maxPt = *itPoint;
				flag = true;
			}
		}

		if (flag) fingertips.push_back(maxPt);
	}
#endif

	// FIXME [check] >> is it correct?
	const cv::Point v1(hullCenterPoint - palmCenterPoint);
#if defined(__GNUC__)
	std::vector<cv::Point>::iterator it = fingertips.begin();
#else
	std::vector<cv::Point>::const_iterator it = fingertips.begin();
#endif
	while (it != fingertips.end())
	{
		const cv::Point v2(*it - palmCenterPoint);
		if (v1.x * v2.x + v1.y * v2.y < 0.0)
			it = fingertips.erase(it);

		if (it == fingertips.end())
			break;
		++it;
	}
}

struct FingerDistanceComparator
{
	FingerDistanceComparator(const cv::Point &thumb)
	: thumb_(thumb)
	{}

	bool operator()(const cv::Point &lhs, const cv::Point &rhs) const
	{
		const double dist1((lhs.x - thumb_.x) * (lhs.x - thumb_.x) + (lhs.y - thumb_.y) * (lhs.y - thumb_.y));
		const double dist2((rhs.x - thumb_.x) * (rhs.x - thumb_.x) + (rhs.y - thumb_.y) * (rhs.y - thumb_.y));
		return dist1 < dist2;
	}

private:
	const cv::Point &thumb_;
};

size_t findThumb(const std::vector<cv::Point> &fingertips)
{
	size_t thumbIdx = -1;

	if (!fingertips.empty())
	{
		cv::Point center(0, 0);
		for (std::vector<cv::Point>::const_iterator it = fingertips.begin(); it != fingertips.end(); ++it)
			center += *it;

		const size_t &count = fingertips.size();
		center.x /= count;
		center.y /= count;

		thumbIdx = std::distance(fingertips.begin(), std::max_element(fingertips.begin(), fingertips.end(), FingerDistanceComparator(center)));
	}

	return thumbIdx;
}

bool findFingerOrder(std::vector<cv::Point> &fingertips)
{
	const size_t &thumbIdx = findThumb(fingertips);

	if (size_t(-1) != thumbIdx)
	{
		std::sort(fingertips.begin(), fingertips.end(), FingerDistanceComparator(fingertips[thumbIdx]));
		return true;
	}
	else return false;
}

void extractMserOnHand(const cv::Mat &img, const cv::Mat &mask, std::vector<cv::KeyPoint> &keypoints)
{
	cv::Mat masked_img;
	img.copyTo(masked_img, mask);

	// "FAST", "STAR", "SIFT", "SURF", "MSER", "GFTT", "HARRIS"
	// also combined format is supported: feature detector adapter name ("Grid", "Pyramid") + feature detector name (see above), e.g. "GridFAST", "PyramidSTAR", etc.
	cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("MSER");
	if (detector.empty())
	{
		std::cout << "can not create detector of given types" << std::endl;
		return;
	}

	detector->detect(masked_img, keypoints);
}

void drawCoutourSequentially(const std::vector<cv::Point> &contour, const std::string &windowName, cv::Mat &cimg)
{
	size_t k = 0;
	for (std::vector<cv::Point>::const_iterator itPt = contour.begin(); itPt != contour.end(); ++itPt, ++k)
	{
		bool is_drawn = false;
		if (0 == k % 30)
		{
			cv::circle(cimg, *itPt, 5, CV_RGB(255, 0, 0), CV_FILLED);
			is_drawn = true;
		}
		if (10 == k % 30)
		{
			cv::circle(cimg, *itPt, 5, CV_RGB(0, 255, 0), CV_FILLED);
			is_drawn = true;
		}
		if (20 == k % 30)
		{
			cv::circle(cimg, *itPt, 5, CV_RGB(0, 0, 255), CV_FILLED);
			is_drawn = true;
		}
		if (is_drawn)
		{
			cv::imshow(windowName, cimg);
			cv::waitKey(50);
		}
	}
}

struct MaxAreaCompare
{
	bool operator()(const std::vector<cv::Point> &lhs, const std::vector<cv::Point> &rhs) const
	{
		const double area1 = cv::contourArea(cv::Mat(lhs));
		const double area2 = cv::contourArea(cv::Mat(rhs));

		return area1 < area2;
	}
};

bool estimateHandPose(const cv::Mat &img, const cv::Mat &gray, cv::Point &palmCenterPoint, std::vector<cv::Point> &fingertips, const bool is_drawn = false, const std::string &windowName = std::string())
{
	CvMemStorage *storage = cvCreateMemStorage();

	const double &startTime = (double)cv::getTickCount();

	//------------------------------------------------------------------
	// extract edge
	cv::Mat edge;
	{
		const int lowerEdgeThreshold = 20, upperEdgeThreshold = 50;
		cv::blur(gray, edge, cv::Size(3, 3));
		cv::Canny(edge, edge, lowerEdgeThreshold, upperEdgeThreshold, 3, true);
	}

	cv::Mat cedge_img;
	if (is_drawn)
	{
		img.copyTo(cedge_img, edge);
		//cedge_img = cedge_img > 0;

		if (!windowName.empty())
		{
			cv::imshow(windowName, cedge_img);
			cv::waitKey(0);
		}
	}

	//------------------------------------------------------------------
	// find hand contour
	std::vector<std::vector<cv::Point> > contours;
	findHandContour(edge, contours);

	if (contours.empty())
	{
		std::cout << "can't find hand contour" << std::endl;
		return false;
	}

	const size_t maxAreaIdx = std::distance(contours.begin(), std::max_element(contours.begin(), contours.end(), MaxAreaCompare()));
	const std::vector<cv::Point> &maxAreaContour = contours[maxAreaIdx];

	// FIXME [delete] >>
	//{
	//	if (is_drawn && !windowName.empty())
	//	{
	//		cv::Mat tmp_img(cedge_img.clone());
	//		for (size_t i = 0; i < contours.size(); ++i)
	//		{
	//			//drawCoutourSequentially(contours[i], windowName, tmp_img);
	//			cv::drawContours(tmp_img, contours, i, CV_RGB(255, 0, 0));
	//			cv::imshow(windowName, tmp_img);
	//			cv::waitKey(0);
	//		}
	//	}
	//}

	//------------------------------------------------------------------
	// calculate convex hull
	std::vector<cv::Point> convexHull;
	cv::convexHull(cv::Mat(maxAreaContour), convexHull, false);

	if (convexHull.empty())
	{
		std::cout << "can't find any convex hull" << std::endl;
		return false;
	}

	const cv::Scalar &handMeanPt = cv::mean(cv::Mat(maxAreaContour));
	const cv::Scalar &hullMeanPt = cv::mean(cv::Mat(convexHull));

	//------------------------------------------------------------------
	// find hand contour & silhouette using hand contour
	std::vector<std::vector<cv::Point> > silhouetteContours;
	//cv::Point palmCenterPoint;
	if (!contours[maxAreaIdx].empty())
		findHandSilhouette(gray.size(), contours, convexHull, maxAreaIdx, hullMeanPt, silhouetteContours, palmCenterPoint);

	if (silhouetteContours.empty())
	{
		std::cout << "can't find hand silhouette" << std::endl;

		if (is_drawn && !windowName.empty())
		{
			cv::circle(cedge_img, palmCenterPoint, 7, CV_RGB(255, 0, 255), 2, 8, 0);
			cv::imshow(windowName, cedge_img);
			cv::waitKey(0);
		}

		return false;
	}

	const size_t maxAreaSilhouetteIdx = std::distance(silhouetteContours.begin(), std::max_element(silhouetteContours.begin(), silhouetteContours.end(), MaxAreaCompare()));
	const std::vector<cv::Point> &maxAreaSilhouetteContour = silhouetteContours[maxAreaSilhouetteIdx];

	// generate hand silhouette
	cv::Mat silhouette_img(gray.size(), gray.depth(), cv::Scalar::all(0));
	if (!silhouetteContours[maxAreaSilhouetteIdx].empty())
	{
		cv::drawContours(silhouette_img, silhouetteContours, maxAreaSilhouetteIdx, CV_RGB(255, 255, 255), 1, 8);
		cv::floodFill(silhouette_img, palmCenterPoint, CV_RGB(255, 255, 255));
	}

	// display hand silhouette
	if (is_drawn && !windowName.empty())
	{
		cv::imshow(windowName, silhouette_img);
		cv::waitKey(0);
	}

	// draw hand contour & convex hull
	if (is_drawn)
	{
		if (!contours[maxAreaIdx].empty())
		{
			const int maxLevel = 0;
			cv::drawContours(cedge_img, contours, maxAreaIdx, CV_RGB(0, 0, 255), 1, 8, std::vector<cv::Vec4i>(), maxLevel, cv::Point());

			const cv::Point *h = (const cv::Point *)&convexHull[0];
			const size_t num = convexHull.size();
			cv::polylines(cedge_img, (const cv::Point **)&h, (int *)&num, 1, true, CV_RGB(0, 255, 0), 1, 8, 0);
		}

		if (!windowName.empty())
		{
			cv::imshow(windowName, cedge_img);
			cv::waitKey(0);
		}
	}

	//------------------------------------------------------------------
	// find convexity defect
	std::vector<std::vector<CvConvexityDefect> > convexityDefects;
	std::vector<cv::Point> convexityDefectPoints;

	const double distanceThreshold = 2.0;
	const double depthThreshold = 5.0;
	findConvexityDefect(storage, maxAreaSilhouetteContour, convexHull, distanceThreshold, depthThreshold, convexityDefects, convexityDefectPoints);

	const cv::Scalar &defectMeanPt = cv::mean(cv::Mat(convexityDefectPoints));

	// draw convexity defect
	if (is_drawn)
	{
		for (std::vector<std::vector<CvConvexityDefect> >::const_iterator it = convexityDefects.begin(); it != convexityDefects.end(); ++it)
		{
			for (std::vector<CvConvexityDefect>::const_iterator itDefect = it->begin(); itDefect != it->end(); ++itDefect)
			{
				cv::line(cedge_img, cv::Point(itDefect->start->x, itDefect->start->y), cv::Point(itDefect->depth_point->x, itDefect->depth_point->y), CV_RGB(255, 255, 0), 1, 8, 0);
				cv::line(cedge_img, cv::Point(itDefect->end->x, itDefect->end->y), cv::Point(itDefect->depth_point->x, itDefect->depth_point->y), CV_RGB(255, 255, 0), 1, 8, 0);
				cv::circle(cedge_img, cv::Point(itDefect->depth_point->x, itDefect->depth_point->y), 2, CV_RGB(255, 255, 0), CV_FILLED, 8, 0);
			}
		}

		if (!windowName.empty())
		{
			cv::imshow(windowName, cedge_img);
			cv::waitKey(0);
		}
	}

	//------------------------------------------------------------------
	// segment fingers
	std::vector<std::vector<cv::Point> > fingerContours;
	{
		const double distanceThreshold = 0.5;

		if (convexityDefectPoints.size() >= 2)
		{
			std::vector<cv::Point>::const_iterator it = convexityDefectPoints.begin(), itPrev = it;
			++it;
			for (; it != convexityDefectPoints.end(); ++it)
			{
				// FIXME [modify] >> Oops !!! stupid implementation
				// it is better to use index of points in the convexity defects
				// I assume that convexity defect points have been already sorted
				const bool retval = segmentFinger(cedge_img, *itPrev, *it, maxAreaSilhouetteContour, distanceThreshold, fingerContours);

				itPrev = it;
			}

			const bool retval = segmentFinger(cedge_img, *itPrev, convexityDefectPoints.front(), maxAreaSilhouetteContour, distanceThreshold, fingerContours);
		}
	}

	//------------------------------------------------------------------
	// find fingertips
	//std::vector<cv::Point> fingertips;
	{
		const size_t displaceIndex = 15;
		findFingertips(fingerContours, displaceIndex, palmCenterPoint, cv::Point(cvRound(hullMeanPt[0]), cvRound(hullMeanPt[1])), fingertips);
	}

	// draw segmented fingers
	if (is_drawn)
	{
		const size_t &numContour = fingerContours.size();
		for (size_t i = 0; i < numContour; ++i)
		{
			if (!fingerContours[i].empty())
			{
				const int r = std::rand() % 256, g = std::rand() % 256, b = std::rand() % 256;
				cv::drawContours(cedge_img, fingerContours, i, CV_RGB(r, g, b), 2, 8);
			}
		}

		if (!windowName.empty())
		{
			cv::imshow(windowName, cedge_img);
			cv::waitKey(0);
		}
	}

	//------------------------------------------------------------------
	// find finger order
	if (!findFingerOrder(fingertips))
	{
		std::cout << "can't order fingers" << std::endl;
		return false;
	}

	// draw fingertips
	if (is_drawn)
	{
		size_t k = 0;
		for (std::vector<cv::Point>::iterator it = fingertips.begin(); it != fingertips.end(); ++it, ++k)
		{
			if (0 == k) cv::circle(cedge_img, *it, 3, CV_RGB(255, 0, 0), 2, 8, 0);
			if (1 == k) cv::circle(cedge_img, *it, 3, CV_RGB(255, 165, 0), 2, 8, 0);
			if (2 == k) cv::circle(cedge_img, *it, 3, CV_RGB(255, 255, 0), 2, 8, 0);
			if (3 == k) cv::circle(cedge_img, *it, 3, CV_RGB(0, 128, 0), 2, 8, 0);
			if (4 == k) cv::circle(cedge_img, *it, 3, CV_RGB(0, 0, 255), 2, 8, 0);
			if (5 == k) cv::circle(cedge_img, *it, 3, CV_RGB(75, 0, 130), 2, 8, 0);
			if (6 == k) cv::circle(cedge_img, *it, 3, CV_RGB(238, 130, 238), 2, 8, 0);
		}
	}

	//------------------------------------------------------------------
	// draw center points
	if (is_drawn)
	{
		cv::circle(cedge_img, cv::Point(cvRound(handMeanPt[0]), cvRound(handMeanPt[1])), 5, CV_RGB(0, 0, 255), 2, 8, 0);
		cv::circle(cedge_img, cv::Point(cvRound(hullMeanPt[0]), cvRound(hullMeanPt[1])), 5, CV_RGB(0, 255, 0), 2, 8, 0);
		cv::circle(cedge_img, cv::Point(cvRound(defectMeanPt[0]), cvRound(defectMeanPt[1])), 5, CV_RGB(255, 0, 0), 2, 8, 0);
		cv::circle(cedge_img, palmCenterPoint, 7, CV_RGB(255, 0, 255), 2, 8, 0);

		if (!windowName.empty())
		{
			cv::imshow(windowName, cedge_img);
			cv::waitKey(0);
		}
	}

	//
	const double &endTime = (double)cv::getTickCount();
	std::cout << "processing time " << ((endTime - startTime) / ((double)cv::getTickFrequency() * 1000.0)) << " ms" << std::endl;
/*
	//------------------------------------------------------------------
	// extract MSER on the hand only
	std::vector<cv::KeyPoint> keypoints;
	extractMserOnHand(img, silhouette_img, keypoints);

	// draw keypoints
	if (is_drawn)
	{
		for (std::vector<cv::KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); ++it)
		{
			const double radius = it->size * 0.1;
			const double angle = it->angle * CV_PI / 180.0;
			cv::circle(cedge_img, cv::Point(cvRound(it->pt.x), cvRound(it->pt.y)), cvRound(radius), CV_RGB(0, 255, 255), 2, 8, 0);
			if (-1 != it->angle)
				cv::line(cedge_img, cv::Point(cvRound(it->pt.x), cvRound(it->pt.y)), cv::Point(cvRound(it->pt.x + radius * std::cos(angle)), cvRound(it->pt.y + radius * std::sin(angle))), CV_RGB(0, 255, 255), 2, 8, 0);
		}

		if (!windowName.empty())
		{
			cv::imshow(windowName, cedge_img);
			cv::waitKey(0);
		}
	}
*/
	//------------------------------------------------------------------
	if (is_drawn && !windowName.empty())
	{
		//
		cv::Mat resultant_img;
		resultant_img = cedge_img;
		//resultant_img = distance_img;
		//resultant_img = silhouette_img;
		cv::imshow(windowName, resultant_img);
	}

	cvClearMemStorage(storage);
	cvReleaseMemStorage(&storage);

	return true;
}

}  // namespace local
}  // unnamed namespace

void hand_pose_estimation()
{
#if 0
	std::list<std::string> filenames;
#if 0
	filenames.push_back("opencv_data\\hand_01.jpg");
	filenames.push_back("opencv_data\\hand_02.jpg");
	filenames.push_back("opencv_data\\hand_03.jpg");
	filenames.push_back("opencv_data\\hand_04.jpg");
	filenames.push_back("opencv_data\\hand_05.jpg");
	filenames.push_back("opencv_data\\hand_06.jpg");
	filenames.push_back("opencv_data\\hand_07.jpg");
	filenames.push_back("opencv_data\\hand_08.jpg");
	filenames.push_back("opencv_data\\hand_09.jpg");
	filenames.push_back("opencv_data\\hand_10.jpg");
	filenames.push_back("opencv_data\\hand_11.jpg");
	filenames.push_back("opencv_data\\hand_12.jpg");
	filenames.push_back("opencv_data\\hand_13.jpg");
	filenames.push_back("opencv_data\\hand_14.jpg");
	filenames.push_back("opencv_data\\hand_15.jpg");
	filenames.push_back("opencv_data\\hand_16.jpg");
	filenames.push_back("opencv_data\\hand_17.jpg");
	filenames.push_back("opencv_data\\hand_18.jpg");
	filenames.push_back("opencv_data\\hand_19.jpg");
	filenames.push_back("opencv_data\\hand_20.jpg");
	filenames.push_back("opencv_data\\hand_21.jpg");
	filenames.push_back("opencv_data\\hand_22.jpg");
	filenames.push_back("opencv_data\\hand_23.jpg");
	filenames.push_back("opencv_data\\hand_24.jpg");
	filenames.push_back("opencv_data\\hand_25.jpg");
	filenames.push_back("opencv_data\\hand_26.jpg");
	filenames.push_back("opencv_data\\hand_27.jpg");
	filenames.push_back("opencv_data\\hand_28.jpg");
	filenames.push_back("opencv_data\\hand_29.jpg");
	filenames.push_back("opencv_data\\hand_30.jpg");
	filenames.push_back("opencv_data\\hand_31.jpg");
	filenames.push_back("opencv_data\\hand_32.jpg");
	filenames.push_back("opencv_data\\hand_33.jpg");
	filenames.push_back("opencv_data\\hand_34.jpg");
	filenames.push_back("opencv_data\\hand_35.jpg");
	filenames.push_back("opencv_data\\hand_36.jpg");
#elif 1
	filenames.push_back("opencv_data\\simple_hand_01.jpg");
	filenames.push_back("opencv_data\\simple_hand_02.jpg");
	filenames.push_back("opencv_data\\simple_hand_03.jpg");
	filenames.push_back("opencv_data\\simple_hand_04.jpg");
	filenames.push_back("opencv_data\\simple_hand_05.jpg");
	filenames.push_back("opencv_data\\simple_hand_06.jpg");
	filenames.push_back("opencv_data\\simple_hand_07.jpg");
	filenames.push_back("opencv_data\\simple_hand_08.jpg");
	filenames.push_back("opencv_data\\simple_hand_09.jpg");
	filenames.push_back("opencv_data\\simple_hand_10.jpg");
	filenames.push_back("opencv_data\\simple_hand_11.jpg");
	filenames.push_back("opencv_data\\simple_hand_12.jpg");
	filenames.push_back("opencv_data\\simple_hand_13.jpg");
#endif

	const char *windowName = "hand pose estimation";
	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);

	//
	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {

		IplImage *srcImage = cvLoadImage(it->c_str());
		if (NULL == srcImage)
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		IplImage *grayImage = NULL;
		if (1 == srcImage->nChannels)
			cvCopy(srcImage, grayImage, NULL);
		else
		{
			grayImage = cvCreateImage(cvGetSize(srcImage), srcImage->depth, 1);
			if (_stricmp(srcImage->channelSeq, "RGB") == 0)
				cvCvtColor(srcImage, grayImage, CV_RGB2GRAY);
			else if (_stricmp(srcImage->channelSeq, "BGR") == 0)
				cvCvtColor(srcImage, grayImage, CV_BGR2GRAY);
			else
				assert(false);
			grayImage->origin = srcImage->origin;
		}

		//
		//local::contour(srcImage, grayImage);
		//local::snake(srcImage, grayImage);
		local::mser(srcImage, grayImage);

		//
		cvShowImage(windowName, srcImage);

		const unsigned char key = cvWaitKey(0);
		if (27 == key)
			break;

		//
		cvReleaseImage(&grayImage);
		cvReleaseImage(&srcImage);
	}

	cvDestroyWindow(windowName);
#elif 1
	std::list<std::string> filenames;
#if 0
	filenames.push_back("opencv_data\\pic1.png");
	filenames.push_back("opencv_data\\pic2.png");
	filenames.push_back("opencv_data\\pic3.png");
	filenames.push_back("opencv_data\\pic4.png");
	filenames.push_back("opencv_data\\pic5.png");
	filenames.push_back("opencv_data\\pic6.png");
	filenames.push_back("opencv_data\\stuff.jpg");
	filenames.push_back("opencv_data\\synthetic_face.png");
	filenames.push_back("opencv_data\\puzzle.png");
	filenames.push_back("opencv_data\\fruits.jpg");
	filenames.push_back("opencv_data\\lena_rgb.bmp");
	filenames.push_back("opencv_data\\hand_01.jpg");
	filenames.push_back("opencv_data\\hand_05.jpg");
	filenames.push_back("opencv_data\\hand_24.jpg");
#elif 1
	//filenames.push_back("opencv_data\\hand_left_1.jpg");
	//filenames.push_back("opencv_data\\hand_right_1.jpg");

	//filenames.push_back("opencv_data\\hand_01.jpg");
	//filenames.push_back("opencv_data\\hand_02.jpg");
	//filenames.push_back("opencv_data\\hand_03.jpg");
	//filenames.push_back("opencv_data\\hand_04.jpg");
	//filenames.push_back("opencv_data\\hand_05.jpg");
	//filenames.push_back("opencv_data\\hand_06.jpg");
	//filenames.push_back("opencv_data\\hand_07.jpg");
	//filenames.push_back("opencv_data\\hand_08.jpg");
	//filenames.push_back("opencv_data\\hand_09.jpg");
	//filenames.push_back("opencv_data\\hand_10.jpg");
	//filenames.push_back("opencv_data\\hand_11.jpg");
	//filenames.push_back("opencv_data\\hand_12.jpg");
	//filenames.push_back("opencv_data\\hand_13.jpg");
	//filenames.push_back("opencv_data\\hand_14.jpg");
	//filenames.push_back("opencv_data\\hand_15.jpg");
	//filenames.push_back("opencv_data\\hand_16.jpg");
	//filenames.push_back("opencv_data\\hand_17.jpg");
	//filenames.push_back("opencv_data\\hand_18.jpg");
	//filenames.push_back("opencv_data\\hand_19.jpg");
	//filenames.push_back("opencv_data\\hand_20.jpg");
	//filenames.push_back("opencv_data\\hand_21.jpg");
	//filenames.push_back("opencv_data\\hand_22.jpg");
	//filenames.push_back("opencv_data\\hand_23.jpg");
	//filenames.push_back("opencv_data\\hand_24.jpg");
	//filenames.push_back("opencv_data\\hand_25.jpg");
	//filenames.push_back("opencv_data\\hand_26.jpg");
	//filenames.push_back("opencv_data\\hand_27.jpg");
	//filenames.push_back("opencv_data\\hand_28.jpg");
	//filenames.push_back("opencv_data\\hand_29.jpg");
	//filenames.push_back("opencv_data\\hand_30.jpg");
	filenames.push_back("opencv_data\\hand_31.jpg");
	filenames.push_back("opencv_data\\hand_32.jpg");
	filenames.push_back("opencv_data\\hand_33.jpg");
	filenames.push_back("opencv_data\\hand_34.jpg");
	filenames.push_back("opencv_data\\hand_35.jpg");
	filenames.push_back("opencv_data\\hand_36.jpg");
#elif 0
	filenames.push_back("opencv_data\\simple_hand_01.jpg");
	filenames.push_back("opencv_data\\simple_hand_02.jpg");
	filenames.push_back("opencv_data\\simple_hand_03.jpg");
	filenames.push_back("opencv_data\\simple_hand_04.jpg");
	filenames.push_back("opencv_data\\simple_hand_05.jpg");
	filenames.push_back("opencv_data\\simple_hand_06.jpg");
	filenames.push_back("opencv_data\\simple_hand_07.jpg");
	filenames.push_back("opencv_data\\simple_hand_08.jpg");
	filenames.push_back("opencv_data\\simple_hand_09.jpg");
	filenames.push_back("opencv_data\\simple_hand_10.jpg");
	filenames.push_back("opencv_data\\simple_hand_11.jpg");
	filenames.push_back("opencv_data\\simple_hand_12.jpg");
	filenames.push_back("opencv_data\\simple_hand_13.jpg");
#endif

	const std::string windowName1("hand pose estimation - original");
	const std::string windowName2("hand pose estimation - processed");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		const cv::Mat &img = cv::imread(*it, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		cv::imshow(windowName1, img);

		cv::Mat gray;
		if (1 == img.channels())
			img.copyTo(gray);
		else
			cv::cvtColor(img, gray, CV_BGR2GRAY);
			//cv::cvtColor(img, gray, CV_RGB2GRAY);

		cv::Point palmCenterPoint;
		std::vector<cv::Point> fingertips;
		if (local::estimateHandPose(img, gray, palmCenterPoint, fingertips, true, windowName2))
		{
			std::cout << "\tthe center of palm: (" << palmCenterPoint.x << ", " << palmCenterPoint.y << ")" << std::endl;
			std::cout << '\t' << fingertips.size() << " fingertips found !!!" << std::endl;
		}

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
#else
	const int imageWidth = 640, imageHeight = 480;

	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (!capture.isOpened())
	{
		std::cout << "fail to open vision sensor" << std::endl;
		return;
	}

	const bool b1 = capture.set(CV_CAP_PROP_FRAME_WIDTH, imageWidth);
	const bool b2 = capture.set(CV_CAP_PROP_FRAME_HEIGHT, imageHeight);

	const std::string windowName("hand pose estimation");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	cv::Mat prevgray, gray, frame, frame2;
	cv::Mat mhi, segmentMask;
	cv::Mat img, blurred;
	std::vector<std::vector<cv::Point> > pointSets;
	std::vector<cv::Vec4i> hierarchy;
	const int maxLevel = 5;
	size_t maxAreaIdx = -1;
	for (;;)
	{
#if defined(_DEBUG) || defined(DEBUG)
		const size_t MAX_SEGMENTATION_FRAMES = 20;
#else
		const size_t MAX_SEGMENTATION_FRAMES = 50;
#endif
		size_t numSegmentationFrames = 0;
		do
		{
			maxAreaIdx = -1;

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
#if 0
			// down-scale and up-scale the image to filter out the noise
			cv::pyrDown(gray, blurred);
			cv::pyrUp(blurred, gray);
#elif 0
			blurred = gray;
			cv::boxFilter(blurred, gray, blurred.type(), cv::Size(5, 5));
#endif

			if  (!prevgray.empty())
			{
				if (mhi.empty())
					mhi.create(gray.rows, gray.cols, CV_32F);

				const bool useConvexHull = false;
				local::segment_motion_using_mhi(useConvexHull, prevgray, gray, mhi, segmentMask, pointSets, hierarchy);
				//local::segment_motion_using_Farneback_motion_estimation(useConvexHull, prevgray, gray, segmentMask, pointSets, hierarchy);

				double maxArea = 0.0;
				size_t idx = 0;
				for (std::vector<std::vector<cv::Point> >::iterator it = pointSets.begin(); it != pointSets.end(); ++it, ++idx)
				{
					if (it->empty()) continue;

					const double area = cv::contourArea(cv::Mat(*it));
					if (area > maxArea)
					{
						maxArea = area;
						maxAreaIdx = idx;
					}
				}

#if 0
				cv::drawContours(img, pointSets, -1, CV_RGB(255, 0, 0), 1, 8, hierarchy, maxLevel, cv::Point());
#elif 0
				const size_t num = pointSets.size();
				for (size_t k = 0; k < num; ++k)
				{
					if (cv::contourArea(cv::Mat(pointSets[k])) < 100.0) continue;
					const int r = rand() % 256, g = rand() % 256, b = rand() % 256;
					if (!pointSets[k].empty()) cv::drawContours(img, pointSets, k, CV_RGB(r, g, b), 1, 8, hierarchy, maxLevel, cv::Point());
				}
#else
				if (-1 != maxAreaIdx)
					if (!pointSets[maxAreaIdx].empty()) cv::drawContours(img, pointSets, maxAreaIdx, CV_RGB(0, 0, 255), 1, 8, hierarchy, 0, cv::Point());
					//if (!pointSets[maxAreaIdx].empty()) cv::drawContours(img, pointSets, maxAreaIdx, CV_RGB(0, 0, 255), 1, 8, hierarchy, maxLevel, cv::Point());
#endif

				std::ostringstream sstream;
				sstream << "segmentation step: " << (numSegmentationFrames + 1) << " / " << MAX_SEGMENTATION_FRAMES;
				cv::putText(img, sstream.str(), cv::Point(5, 10), cv::FONT_HERSHEY_COMPLEX, 0.3, CV_RGB(255, 0, 255), 1, 8, false);
				cv::imshow(windowName, img);
			}

			if (cv::waitKey(1) >= 0)
				break;

			std::swap(prevgray, gray);
		} while (++numSegmentationFrames < MAX_SEGMENTATION_FRAMES);

		//
#if defined(_DEBUG) || defined(DEBUG)
		const size_t MAX_ACQUISITION_FRAMES = 5;
#else
		const size_t MAX_ACQUISITION_FRAMES = 20;
#endif
		size_t numAcquisitionFrames = 0;
		do
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

			if (-1 != maxAreaIdx)
				if (!pointSets[maxAreaIdx].empty()) cv::drawContours(img, pointSets, maxAreaIdx, CV_RGB(0, 0, 255), 1, 8, hierarchy, 0, cv::Point());
				//if (!pointSets[maxAreaIdx].empty()) cv::drawContours(img, pointSets, maxAreaIdx, CV_RGB(0, 0, 255), 1, 8, hierarchy, maxLevel, cv::Point());

			std::ostringstream sstream;
			sstream << "acquisition step: " << (numAcquisitionFrames + 1) << " / " << MAX_ACQUISITION_FRAMES;
			cv::putText(img, sstream.str(), cv::Point(5, 10), cv::FONT_HERSHEY_COMPLEX, 0.3, CV_RGB(255, 0, 255), 1, 8, false);
			cv::imshow(windowName, img);

			if (cv::waitKey(30) >= 0)
				break;
		} while (++numAcquisitionFrames < MAX_ACQUISITION_FRAMES);

		//
		const size_t NUMBER_OF_SNAKE_POINTS = 0;
		if (!pointSets.empty() && -1 != maxAreaIdx)
		{
			{
				const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));
				const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
				const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
				cv::dilate(segmentMask, segmentMask, selement3, cv::Point(-1, -1), 3);
				cv::erode(segmentMask, segmentMask, selement3, cv::Point(-1, -1), 5);
			}

			cv::Mat semgented_gray;
			gray.copyTo(semgented_gray, segmentMask);
			//cv::equalizeHist(semgented_gray, semgented_gray);

			// FIXME [delete] >>
			cv::Mat semgented_img;
			img.copyTo(semgented_img, segmentMask);

			std::vector<cv::Point> snake_contour;
			local::fit_contour_by_snake(semgented_gray, pointSets[maxAreaIdx], NUMBER_OF_SNAKE_POINTS, snake_contour);

			if (!snake_contour.empty())
			{
				// draw snake on image
				const cv::Point *snake_pts = &snake_contour[0];
				const size_t numSnakePts = 0 == NUMBER_OF_SNAKE_POINTS ? snake_contour.size() : NUMBER_OF_SNAKE_POINTS;
				cv::polylines(semgented_img, (const cv::Point **)&snake_pts, (int *)&numSnakePts, 1, true, CV_RGB(255, 0, 0), 2, 8, 0);

				cv::putText(semgented_img, "pose estimation step", cv::Point(5, 10), cv::FONT_HERSHEY_COMPLEX, 0.3, CV_RGB(255, 0, 255), 1, 8, false);
				cv::imshow(windowName, semgented_img);
			}
		}

		if (cv::waitKey(0) >= 0)
			break;
	}

	cv::destroyWindow(windowName);
#endif
}
