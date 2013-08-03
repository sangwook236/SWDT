//#include "stdafx.h"
//#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/legacy/compat.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iterator>
#include <list>
#include <limits>
#include <ctime>


namespace my_opencv {

void contour(IplImage *srcImg, IplImage *grayImg);
void make_contour(const cv::Mat &img, const cv::Rect &roi, const int segmentId, std::vector<std::vector<cv::Point> > &contours, std::vector<cv::Vec4i> &hierarchy);
void make_convex_hull(const cv::Mat &img, const cv::Rect &roi, const int segmentId, std::vector<cv::Point> &convexHull);
void find_convexity_defect(CvMemStorage *storage, const std::vector<cv::Point> &contour, const std::vector<cv::Point> &convexHull, const double distanceThreshold, const double depthThreshold, std::vector<std::vector<CvConvexityDefect> > &convexityDefects, std::vector<cv::Point> &convexityDefectPoints);
bool calculate_curvature(const cv::Point2d &v1, const cv::Point2d &v2, double &curvature);
bool find_curvature_points(const std::vector<cv::Point> &fingerContour, const size_t displaceIndex, size_t &minIdx, size_t &maxIdx);
void compute_distance_transform(const cv::Mat &gray, cv::Mat &distanceTransform);

void snake(IplImage *srcImage, IplImage *grayImage);

void segment_motion_using_mhi(const bool useConvexHull, const cv::Mat &prev_gray_img, const cv::Mat &curr_gray_img, cv::Mat &mhi, cv::Mat &segmentMask, std::vector<std::vector<cv::Point> > &pointSets, std::vector<cv::Vec4i> &hierarchy);

}  // namespace my_opencv

namespace {
namespace local {

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
		my_opencv::make_convex_hull(segmentMask, cv::Rect(), 1, convexHull);
		if (!convexHull.empty()) pointSets.push_back(convexHull);
	}
	else
	{
		my_opencv::make_contour(segmentMask, cv::Rect(), 1, pointSets, hierarchy);
	}
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
	my_opencv::compute_distance_transform(gray, distanceTransform);

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
		if (my_opencv::find_curvature_points(*it, displaceIndex, minIdx, maxIdx))
		{
			if ((size_t)-1 != minIdx) minFingerTipPoints.push_back((*it)[minIdx]);
			if ((size_t)-1 != maxIdx) maxFingerTipPoints.push_back((*it)[maxIdx]);
		}
	}

	fingertips.swap(minFingerTipPoints);
#elif 0
	const size_t &count = fingerContours.size();
	fingertips.reserve(count);
	for (std::vector<std::vector<cv::Point> >::const_iterator it = fingerContours.begin(); it != fingerContours.end(); ++it)
	{
		size_t minIdx = -1, maxIdx = -1;
		if (find_curvature_points(*it, displaceIndex, minIdx, maxIdx))
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
	my_opencv::find_convexity_defect(storage, maxAreaSilhouetteContour, convexHull, distanceThreshold, depthThreshold, convexityDefects, convexityDefectPoints);

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

namespace my_opencv {

void hand_pose_estimation()
{
#if 0
	std::list<std::string> filenames;
#if 0
	filenames.push_back("./data/machine_vision/opencv/hand_01.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_02.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_03.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_04.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_05.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_06.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_07.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_08.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_09.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_10.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_11.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_12.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_13.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_14.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_15.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_16.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_17.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_18.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_19.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_20.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_21.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_22.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_23.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_24.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_25.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_26.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_27.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_28.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_29.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_30.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_31.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_32.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_33.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_34.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_35.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_36.jpg");
#elif 1
	filenames.push_back("./data/machine_vision/opencv/simple_hand_01.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_02.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_03.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_04.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_05.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_06.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_07.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_08.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_09.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_10.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_11.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_12.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_13.jpg");
#endif

	const char *windowName = "hand pose estimation";
	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);

	//
	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {

		IplImage *srcImage = cvLoadImage(it->c_str());
		if (NULL == srcImage)
		{
			std::cout << "image file not found: " << *it << std::endl;
			continue;
		}

		IplImage *grayImage = NULL;
		if (1 == srcImage->nChannels)
			cvCopy(srcImage, grayImage, NULL);
		else
		{
			grayImage = cvCreateImage(cvGetSize(srcImage), srcImage->depth, 1);
#if defined(__GNUC__)
			if (strcasecmp(image->channelSeq, "RGB") == 0)
#elif defined(_MSC_VER)
			if (_stricmp(srcImage->channelSeq, "RGB") == 0)
#endif
				cvCvtColor(srcImage, grayImage, CV_RGB2GRAY);
#if defined(__GNUC__)
			if (strcasecmp(image->channelSeq, "BGR") == 0)
#elif defined(_MSC_VER)
			else if (_stricmp(srcImage->channelSeq, "BGR") == 0)
#endif
				cvCvtColor(srcImage, grayImage, CV_BGR2GRAY);
			else
				assert(false);
			grayImage->origin = srcImage->origin;
		}

		// smoothing.
		// TODO [check] >> smoothing is needed?

		//
		//contour(srcImage, grayImage);
		//snake(srcImage, grayImage);
#if 0
		local::mser(srcImage, grayImage);
#else
		local::mser(cv::Mat(srcImage), cv::Mat(grayImage));
#endif

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
	filenames.push_back("./data/machine_vision/opencv/pic1.png");
	filenames.push_back("./data/machine_vision/opencv/pic2.png");
	filenames.push_back("./data/machine_vision/opencv/pic3.png");
	filenames.push_back("./data/machine_vision/opencv/pic4.png");
	filenames.push_back("./data/machine_vision/opencv/pic5.png");
	filenames.push_back("./data/machine_vision/opencv/pic6.png");
	filenames.push_back("./data/machine_vision/opencv/stuff.jpg");
	filenames.push_back("./data/machine_vision/opencv/synthetic_face.png");
	filenames.push_back("./data/machine_vision/opencv/puzzle.png");
	filenames.push_back("./data/machine_vision/opencv/fruits.jpg");
	filenames.push_back("./data/machine_vision/opencv/lena_rgb.bmp");
	filenames.push_back("./data/machine_vision/opencv/hand_01.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_05.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_24.jpg");
#elif 1
	//filenames.push_back("./data/machine_vision/opencv/hand_left_1.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_right_1.jpg");

	//filenames.push_back("./data/machine_vision/opencv/hand_01.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_02.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_03.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_04.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_05.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_06.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_07.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_08.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_09.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_10.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_11.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_12.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_13.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_14.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_15.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_16.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_17.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_18.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_19.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_20.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_21.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_22.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_23.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_24.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_25.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_26.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_27.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_28.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_29.jpg");
	//filenames.push_back("./data/machine_vision/opencv/hand_30.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_31.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_32.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_33.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_34.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_35.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_36.jpg");
#elif 0
	filenames.push_back("./data/machine_vision/opencv/simple_hand_01.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_02.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_03.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_04.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_05.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_06.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_07.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_08.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_09.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_10.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_11.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_12.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_13.jpg");
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
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

	const bool b1 = capture.set(CV_CAP_PROP_FRAME_WIDTH, imageWidth);
	const bool b2 = capture.set(CV_CAP_PROP_FRAME_HEIGHT, imageHeight);

	const std::string windowName("hand pose estimation");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	cv::Mat prevgray, gray, frame, frame2;
	cv::Mat mhi, segmentMask;
	cv::Mat img;
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

			cv::cvtColor(frame, gray, CV_BGR2GRAY);
			cv::cvtColor(gray, img, CV_GRAY2BGR);

			// smoothing
#if 0
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

			if  (!prevgray.empty())
			{
				if (mhi.empty())
					mhi.create(gray.rows, gray.cols, CV_32F);

				const bool useConvexHull = false;
				segment_motion_using_mhi(useConvexHull, prevgray, gray, mhi, segmentMask, pointSets, hierarchy);
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
					if (!pointSets[k].empty())
						cv::drawContours(img, pointSets, k, CV_RGB(r, g, b), 1, 8, hierarchy, maxLevel, cv::Point());
				}
#else
				if (-1 != maxAreaIdx)
					if (!pointSets[maxAreaIdx].empty())
					{
						cv::drawContours(img, pointSets, maxAreaIdx, CV_RGB(0, 0, 255), 1, 8, hierarchy, 0, cv::Point());
						//cv::drawContours(img, pointSets, maxAreaIdx, CV_RGB(0, 0, 255), 1, 8, hierarchy, maxLevel, cv::Point());
					}
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

		if (!pointSets.empty() && -1 != maxAreaIdx)
		{
			{
				const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));
				const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
				const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
#if 0
				cv::dilate(segmentMask, segmentMask, selement3, cv::Point(-1, -1), 3);
				cv::erode(segmentMask, segmentMask, selement3, cv::Point(-1, -1), 5);
#else
				cv::morphologyEx(segmentMask, segmentMask, cv::MORPH_CLOSE, selement3, cv::Point(-1, -1), 3);
				cv::morphologyEx(segmentMask, segmentMask, cv::MORPH_OPEN, selement3, cv::Point(-1, -1), 5);
#endif
			}

			cv::Mat semgented_gray;
			gray.copyTo(semgented_gray, segmentMask);
			//cv::equalizeHist(semgented_gray, semgented_gray);

			// FIXME [delete] >>
			cv::Mat semgented_img;
			img.copyTo(semgented_img, segmentMask);

			//const size_t NUMBER_OF_SNAKE_POINTS = 50;
			const size_t NUMBER_OF_SNAKE_POINTS = 0;
			const float alpha = 3.0f;  // weight(s) of continuity energy, single float or array of length floats, one for each contour point.
			const float beta = 5.0f;  // weight(s) of curvature energy, single float or array of length floats, one for each contour point.
			const float gamma = 2.0f;  // weight(s) of image energy, single float or array of length floats, one for each contour point.
			const bool use_gradient = true;  // gradient flag; if true, the function calculates the gradient magnitude for every image pixel and consideres it as the energy field, otherwise the input image itself is considered.
			const CvSize win = cvSize(21, 21);  // size of neighborhood of every point used to search the minimum, both win.width and win.height must be odd.
			std::vector<cv::Point> snake_contour;
			local::fit_contour_by_snake(semgented_gray, pointSets[maxAreaIdx], NUMBER_OF_SNAKE_POINTS, alpha, beta, gamma, use_gradient, win, snake_contour);

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

}  // namespace my_opencv
