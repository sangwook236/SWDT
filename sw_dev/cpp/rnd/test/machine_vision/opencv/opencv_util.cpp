//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
//#include <opencv2/legacy/compat.hpp>
//#include <opencv2/legacy/legacy.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/opencv.hpp>
#include <boost/polygon/polygon.hpp>
#include <set>
#include <iterator>
#include <stdexcept>
#include <ctime>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

cv::Rect get_bounding_box(const std::vector<cv::Point> &points)
{
	// [ref]
	//	cv::RotatedRect::boundingRect()
	//	cv::minAreaRect()
	//	cv::minEnclosingCircle()
	//	cv::boundingRect()

	typedef boost::polygon::polygon_data<int> Polygon;
	typedef boost::polygon::polygon_traits<Polygon>::point_type Point;

	std::vector<Point> pts;
	pts.reserve(points.size());
	for (std::vector<cv::Point>::const_iterator it = points.begin(); it != points.end(); ++it)
		pts.push_back(Point(it->x, it->y));

	Polygon poly;
	boost::polygon::set_points(poly, pts.begin(), pts.end());

	boost::polygon::rectangle_data<int> rect;
	boost::polygon::extents(rect, poly);

	//const int xl = boost::polygon::xl(rect);
	//const int xh = boost::polygon::xh(rect);
	//const int yl = boost::polygon::yl(rect);
	//const int yh = boost::polygon::yh(rect);
	//const boost::polygon::rectangle_data<int>::interval_type h = boost::polygon::horizontal(rect);
	//const boost::polygon::rectangle_data<int>::interval_type v = boost::polygon::vertical(rect);

	return cv::Rect(boost::polygon::xl(rect), boost::polygon::yl(rect), boost::polygon::xh(rect) - boost::polygon::xl(rect), boost::polygon::yh(rect) - boost::polygon::yl(rect));
}

std::set<int> unique(const cv::Mat& input)
{
	typedef int data_type;

	if (input.channels() > 1 || input.type() != CV_32S)
		//if (input.channels() > 1 || input.type() != CV_32F)
	{
		std::cerr << "unique !!! Only works with CV_32F 1-channel Mat" << std::endl;
		return std::set<data_type>();
	}

	std::set<data_type> out;
	for (int y = 0; y < input.rows; ++y)
	{
		const data_type* row_ptr = input.ptr<data_type>(y);
		for (int x = 0; x < input.cols; ++x)
		{
			const data_type& value = row_ptr[x];

			if (out.find(value) == out.end())
				out.insert(value);
		}
	}

	return out;
}

void canny(const cv::Mat &gray, const int lowerEdgeThreshold, const int upperEdgeThreshold, const bool useL2, cv::Mat &edge)
{
	// smoothing
#if 0
	// METHOD #1: down-scale and up-scale the image to filter out the noise.

	{
		cv::Mat tmp;
		cv::pyrDown(gray, tmp);
		cv::pyrUp(tmp, edge);
	}
#elif 0
	// METHOD #2: Gaussian filtering.

	{
		// FIXME [adjust] >> adjust parameters.
		const int kernelSize = 3;
		const double sigma = 2.0;
		cv::GaussianBlur(gray, edge, cv::Size(kernelSize, kernelSize), sigma, sigma, cv::BORDER_DEFAULT);
	}
#elif 1
	// METHOD #3: box filtering.

	{
		// FIXME [adjust] >> adjust parameters.
		const int ddepth = -1;  // the output image depth. -1 to use src.depth().
		const int kernelSize = 3;
		const bool normalize = true;
		cv::boxFilter(gray, edge, ddepth, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), normalize, cv::BORDER_DEFAULT);
		//cv::blur(gray, edge, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), cv::BORDER_DEFAULT);  // use the normalized box filter.
	}
#elif 0
	// METHOD #4: bilateral filtering.

	{
		// FIXME [adjust] >> adjust parameters.
		const int diameter = -1;  // diameter of each pixel neighborhood that is used during filtering. if it is non-positive, it is computed from sigmaSpace.
		const double sigmaColor = 3.0;  // for range filter.
		const double sigmaSpace = 50.0;  // for space filter.
		cv::bilateralFilter(gray, edge, diameter, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);
	}
#else
	// METHOD #5: no filtering.

	edge = gray;
#endif

	// run the edge detector on grayscale
	const int apertureSize = 3;  // aperture size for the Sobel() operator.
	cv::Canny(edge, edge, lowerEdgeThreshold, upperEdgeThreshold, apertureSize, useL2);
}

void sobel(const cv::Mat &gray, const double thresholdRatio, cv::Mat &gradient)
{
	//const int ksize = 5;
	const int ksize = CV_SCHARR;
	cv::Mat xgradient, ygradient;

	cv::Sobel(gray, xgradient, CV_32FC1, 1, 0, ksize, 1.0, 0.0, cv::BORDER_DEFAULT);
	cv::Sobel(gray, ygradient, CV_32FC1, 0, 1, ksize, 1.0, 0.0, cv::BORDER_DEFAULT);

	cv::magnitude(xgradient, ygradient, gradient);

	if (0.0 < thresholdRatio && thresholdRatio <= 1.0)
	{
		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(gradient, &minVal, &maxVal);
		gradient = gradient >= (minVal + (maxVal - minVal) * thresholdRatio);
		//cv::compare(gradient, minVal + (maxVal - minVal) * thresholdRatio, gradient, cv::CMP_GT);
	}
}

void dilation(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations)
{
	cv::dilate(src, dst, selement, cv::Point(-1, -1), iterations);
}

void erosion(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations)
{
	cv::erode(src, dst, selement, cv::Point(-1, -1), iterations);
}

void opening(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations)
{
	// opening = dilation -> erosion
	cv::morphologyEx(src, dst, cv::MORPH_OPEN, selement, cv::Point(-1, -1), iterations);
}

void closing(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations)
{
	// closing = erosion -> dilation
	cv::morphologyEx(src, dst, cv::MORPH_CLOSE, selement, cv::Point(-1, -1), iterations);
}

void gradient(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations)
{
	// gradient = dilation - erosion
	cv::morphologyEx(src, dst, cv::MORPH_GRADIENT, selement, cv::Point(-1, -1), iterations);
}

void hit_and_miss()
{
	throw std::runtime_error("not yet implemented");
}

void top_hat(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations)
{
	// top_hat = src - opening
	cv::morphologyEx(src, dst, cv::MORPH_TOPHAT, selement, cv::Point(-1, -1), iterations);
}

void bottom_hat(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations)
{
	// bottom_hat = closing - src
	cv::morphologyEx(src, dst, cv::MORPH_BLACKHAT, selement, cv::Point(-1, -1), iterations);
}

void compute_distance_transform(const cv::Mat &gray, cv::Mat &distanceTransform)
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

void make_convex_hull(const cv::Mat &img, const cv::Rect &roi, const int pixVal, std::vector<cv::Point> &convexHull)
{
	const cv::Mat &roi_img = roi.width == 0 || roi.height == 0 ? img : img(roi);

	std::vector<cv::Point> points;
	points.reserve(roi_img.rows * roi_img.cols);
	for (int r = 0; r < roi_img.rows; ++r)
		for (int c = 0; c < roi_img.cols; ++c)
		{
			if (roi_img.at<unsigned char>(r, c) == pixVal)
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

bool simple_convex_hull(const cv::Mat &img, const cv::Rect &roi, const int pixVal, std::vector<cv::Point> &convexHull)
{
	const cv::Mat &roi_img = roi.width == 0 || roi.height == 0 ? img : img(roi);

	std::vector<cv::Point> points;
	points.reserve(roi_img.rows * roi_img.cols);
	for (int r = 0; r < roi_img.rows; ++r)
		for (int c = 0; c < roi_img.cols; ++c)
		{
			if (roi_img.at<unsigned char>(r, c) == pixVal)
				points.push_back(cv::Point(roi.x + c, roi.y + r));
		}
	if (points.empty()) return false;

	cv::convexHull(cv::Mat(points), convexHull, false);
	if (convexHull.empty()) return false;

#if 1
    // comment this out if you do not want approximation
	cv::approxPolyDP(convexHull, convexHull, 3.0, true);
#endif

	return true;
}

bool calculate_curvature(const cv::Point2d &v1, const cv::Point2d &v2, double &curvature)
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

bool find_curvature_points(const std::vector<cv::Point> &fingerContour, const size_t displaceIndex, size_t &minIdx, size_t &maxIdx)
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
		if (calculate_curvature(cv::Point2d(pt1.x, pt1.y), cv::Point2d(pt2.x, pt2.y), curvature))
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

void find_convexity_defect(CvMemStorage *storage, const std::vector<cv::Point> &contour, const std::vector<cv::Point> &convexHull, const double distanceThreshold, const double depthThreshold, std::vector<std::vector<CvConvexityDefect> > &convexityDefects, std::vector<cv::Point> &convexityDefectPoints)
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

void draw_histogram_1D(const cv::MatND &hist, const int binCount, const double maxVal, const int binWidth, const int maxHeight, cv::Mat &histImg)
{
#if 0
	for (int i = 0; i < binCount; ++i)
	{
		const float binVal(hist.at<float>(i));
		const int binHeight(cvRound(binVal * maxHeight / maxVal));
		cv::rectangle(
			histImg,
			cv::Point(i*binWidth, maxHeight), cv::Point((i+1)*binWidth - 1, maxHeight - binHeight),
			binVal > maxVal ? CV_RGB(255, 0, 0) : CV_RGB(255, 255, 255),
			CV_FILLED
		);
	}
#else
	const float *binPtr = (const float *)hist.data;
	for (int i = 0; i < binCount; ++i, ++binPtr)
	{
		const int binHeight(cvRound(*binPtr * maxHeight / maxVal));
		cv::rectangle(
			histImg,
			cv::Point(i*binWidth, maxHeight), cv::Point((i+1)*binWidth - 1, maxHeight - binHeight),
			*binPtr > maxVal ? CV_RGB(255, 0, 0) : CV_RGB(255, 255, 255),
			CV_FILLED
		);
	}
#endif
}

void draw_histogram_2D(const cv::MatND &hist, const int horzBinCount, const int vertBinCount, const double maxVal, const int horzBinSize, const int vertBinSize, cv::Mat &histImg)
{
#if 0
	for (int v = 0; v < vertBinCount; ++v)
		for (int h = 0; h < horzBinCount; ++h)
		{
			const float binVal(hist.at<float>(v, h));
			cv::rectangle(
				histImg,
				cv::Point(h*horzBinSize, v*vertBinSize), cv::Point((h+1)*horzBinSize - 1, (v+1)*vertBinSize - 1),
				binVal > maxVal ? CV_RGB(255, 0, 0) : cv::Scalar::all(cvRound(binVal * 255.0 / maxVal)),
				CV_FILLED
			);
		}
#else
	const float *binPtr = (const float *)hist.data;
	for (int v = 0; v < vertBinCount; ++v)
		for (int h = 0; h < horzBinCount; ++h, ++binPtr)
		{
			cv::rectangle(
				histImg,
				cv::Point(h*horzBinSize, v*vertBinSize), cv::Point((h+1)*horzBinSize - 1, (v+1)*vertBinSize - 1),
				*binPtr > maxVal ? cv::Scalar(CV_RGB(255, 0, 0)) : cv::Scalar::all(cvRound(*binPtr * 255.0 / maxVal)),
				CV_FILLED
			);
		}
#endif
}

// the function normalizes the histogram bins by scaling them, such that the sum of the bins becomes equal to factor
void normalize_histogram(cv::MatND &hist, const double factor)
{
#if 0
	// FIXME [modify] >>
	cvNormalizeHist(&(CvHistogram)hist, factor);
#else
	const cv::Scalar sums(cv::sum(hist));

	const double eps = 1.0e-20;
	if (std::fabs(sums[0]) < eps) return;

	//cv::Mat tmp(hist);
	//tmp.convertTo(hist, -1, factor / sums[0], 0.0);
	hist *= factor / sums[0];
#endif
}

#if 0
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

	std::cout << "MSER extracted " << contours->total << " contours in " << (t / ((double)cvGetTickFrequency() * 1000.0)) << " ms" << std::endl;

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
#else
// [ref] ${OPENCV_ROOT}/sample/c/mser_sample.cpp
void mser(cv::Mat &srcImage, const cv::Mat &grayImage)
{
	const cv::Scalar colors[] =
	{
		cv::Scalar(0, 0, 255),
		cv::Scalar(0, 128, 255),
		cv::Scalar(0, 255, 255),
		cv::Scalar(0, 255, 0),
		cv::Scalar(255, 128, 0),
		cv::Scalar(255, 255, 0),
		cv::Scalar(255, 0, 0),
		cv::Scalar(255, 0, 255),
		cv::Scalar(255, 255, 255),
		cv::Scalar(196, 255, 255),
		cv::Scalar(255, 255, 196)
	};

	const cv::Vec3b bcolors[] =
	{
		cv::Vec3b(0, 0, 255),
		cv::Vec3b(0, 128, 255),
		cv::Vec3b(0, 255, 255),
		cv::Vec3b(0, 255, 0),
		cv::Vec3b(255, 128, 0),
		cv::Vec3b(255, 255, 0),
		cv::Vec3b(255, 0, 0),
		cv::Vec3b(255, 0, 255),
		cv::Vec3b(255, 255, 255)
	};

	cv::Mat yuv(srcImage.size(), CV_8UC3);
	cv::cvtColor(srcImage, yuv, CV_BGR2YCrCb);

	const int delta = 5;
	const int min_area = 60;
	const int max_area = 14400;
	const float max_variation = 0.25f;
	const float min_diversity = 0.2f;
	const int max_evolution = 200;
	const double area_threshold = 1.01;
	const double min_margin = 0.003;
	const int edge_blur_size = 5;
	cv::Ptr<cv::MSER> mser = cv::MSER::create();

	double t = (double)cv::getTickCount();
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Rect> bboxes;
	mser->detectRegions(yuv, contours, bboxes);
	t = cv::getTickCount() - t;

	std::cout << "MSER extracted " << contours.size() << " contours in " << (t / ((double)cv::getTickFrequency() * 1000.0)) << " ms" << std::endl;

	// find ellipse ( it seems cvFitEllipse2 have error or sth? )
	// FIXME [check] >> there are some errors. have to compare original source (mser_sample.cpp)
    for (int i = (int)contours.size() - 1; i >= 0; --i)
	{
        const std::vector<cv::Point> &r = contours[i];
        for (int j = 0; j < (int)r.size(); ++j)
        {
            const cv::Point &pt = r[j];
            srcImage.at<cv::Vec3b>(pt) = bcolors[i % 9];
        }

        // find ellipse (it seems cvfitellipse2 have error or sth?)
        cv::RotatedRect box = cv::fitEllipse(r);

        box.angle = (float)CV_PI / 2 - box.angle;
        cv::ellipse(srcImage, box, colors[10], 2);
	}
}
#endif

void snake(IplImage *srcImage, IplImage *grayImage)
{
	const int NUMBER_OF_SNAKE_POINTS = 50;
	const int threshold = 90;

	float alpha = 3.0f;
	float beta = 5.0f;
	float gamma = 2.0f;
	const int use_gradient = 1;
	const CvSize win = cvSize(21, 21);
	const CvTermCriteria term_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, 1.0);

	IplImage *img = cvCloneImage(grayImage);

	{
		IplImage *tmp_img = cvCloneImage(grayImage);

		// make a average filtering.
		cvSmooth(tmp_img, img, CV_BLUR, 31, 15);
		//iplBlur(tmp_img, img, 31, 31, 15, 15);  // don't use IPL.

		// do a threshold.
		cvThreshold(img, tmp_img, threshold, 255, CV_THRESH_BINARY);
		//iplThreshold(img, tmp_img, threshold);  // don't use IPL.

		// expand the thresholded image of ones - smoothing the edge.
		// move start position of snake out since there are no balloon force.
		cvDilate(tmp_img, img, NULL, 3);

		cvReleaseImage(&tmp_img);
	}

	// find the contours.
	CvMemStorage *storage = cvCreateMemStorage(0);
	CvSeq *contour = NULL;
	cvFindContours(img, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	// run through the found coutours.
	std::vector<CvPoint> points(NUMBER_OF_SNAKE_POINTS, cvPoint(0, 0));
	while (contour)
	{
		if (NUMBER_OF_SNAKE_POINTS <= contour->total)
		{
			//memset(points, 0, NUMBER_OF_SNAKE_POINTS * sizeof(CvPoint));

			cvSmooth(grayImage, img, CV_BLUR, 7, 3);
			//iplBlur(grayImage, img, 7, 7, 3, 3);  // don't use IPL.

#if 0
			std::vecto<CvPoint> pts(contour->total);
			cvCvtSeqToArray(contour, &pts[0], CV_WHOLE_SEQ);  // copy the contour to an array.

			// number of jumps between the desired points (downsample only!).
			const int stride = int(contour->total / NUMBER_OF_SNAKE_POINTS);
			for (int i = 0; i < NUMBER_OF_SNAKE_POINTS; ++i)
			{
				points[i].x = pts[int(i * stride)].x;
				points[i].y = pts[int(i * stride)].y;
			}
#else
			const int stride = int(contour->total / NUMBER_OF_SNAKE_POINTS);
			for (int i = 0; i < NUMBER_OF_SNAKE_POINTS; ++i)
			{
				CvPoint *pt = CV_GET_SEQ_ELEM(CvPoint, contour, i * stride);
				points[i].x = pt->x;
				points[i].y = pt->y;
			}
#endif

			// iterate snake.
			// FIXME [correct] >> Compile-time error.
			//cvSnakeImage(img, &points[0], NUMBER_OF_SNAKE_POINTS, &alpha, &beta, &gamma, CV_VALUE, win, term_criteria, use_gradient);

			// draw snake on image.
			CvPoint *points_ptr = (CvPoint *)&points[0];
			cvPolyLine(srcImage, (CvPoint **)points_ptr, &NUMBER_OF_SNAKE_POINTS, 1, 1, CV_RGB(255, 0, 0), 3, 8, 0);
		}

		// get next contours.
		contour = contour->h_next;
	}

	cvReleaseMemStorage(&storage);
	cvReleaseImage(&img);
}

void fit_contour_by_snake(const cv::Mat &gray_img, const std::vector<cv::Point> &contour, const size_t numSnakePoints, const float alpha, const float beta, const float gamma, const bool use_gradient, const CvSize &win, std::vector<cv::Point> &snake_contour)
{
	snake_contour.clear();
	if (contour.empty()) return;

/*
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Point> hierarchy;
	{
		const int threshold = 90;

		cv::Mat binary_img;

		// make a average filtering.
		cv::blur(gray_img, binary_img, cv::Size(31, 15));

		// do a threshold.
		cv::threshold(binary_img, binary_img, threshold, 255, cv::THRESH_BINARY);

		// expand the thressholded image of ones - smoothing the edge.
		// move start position of snake out since there are no ballon force.
		{
			const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
			cv::dilate(binary_img, binary_img, selement, cv::Point(-1, -1), 3);
		}

		cv::findContours(binary_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	}
*/

	const CvTermCriteria term_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, 1.0);

	// run through the found coutours.
	const size_t &numPts = contour.size();
	const size_t numSnakePts = 0 == numSnakePoints ? numPts : numSnakePoints;
	if (numPts >= numSnakePts)
	{
		std::vector<CvPoint> points(numSnakePts, cvPoint(0, 0));

		cv::Mat blurred_img;
		cv::blur(gray_img, blurred_img, cv::Size(7, 3));

		const int stride = int(numPts / numSnakePts);
		for (size_t i = 0; i < numSnakePts; ++i)
		{
			const cv::Point &pt = contour[i * stride];
			points[i] = cvPoint(pt.x, pt.y);
		}

		// iterate snake.
#if defined(__GNUC__)
        IplImage blurred_img_ipl = (IplImage)blurred_img;
		cvSnakeImage(&blurred_img_ipl, &points[0], numSnakePts, (float *)&alpha, (float *)&beta, (float *)&gamma, CV_VALUE, win, term_criteria, use_gradient ? 1 : 0);
#else
		// FIXME [correct] >> Compile-time error.
		//cvSnakeImage(&(IplImage)blurred_img, &points[0], numSnakePts, (float *)&alpha, (float *)&beta, (float *)&gamma, CV_VALUE, win, term_criteria, use_gradient ? 1 : 0);
#endif

		snake_contour.assign(points.begin(), points.end());
	}
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
	cv::motempl::updateMotionHistory(silh, mhi, timestamp, MHI_DURATION);  // update MHI

	//
	cv::Mat processed_mhi;  // processed MHI
	{
		const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));
		const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
		const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
		const int iterations = 1;
#if 0
		cv::erode(mhi, processed_mhi, selement5, cv::Point(-1, -1), iterations);
		cv::dilate(processed_mhi, processed_mhi, selement5, cv::Point(-1, -1), iterations);
#else
		cv::morphologyEx(mhi, processed_mhi, cv::MORPH_OPEN, selement5, cv::Point(-1, -1), iterations);
		cv::morphologyEx(processed_mhi, processed_mhi, cv::MORPH_CLOSE, selement5, cv::Point(-1, -1), iterations);
#endif
	}

	// calculate motion gradient orientation and valid orientation mask
	cv::Mat mask;  // valid orientation mask
	cv::Mat orient;  // orientation
	cv::motempl::calcMotionGradient(processed_mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, motion_gradient_aperture_size);

	// segment motion: get sequence of motion components
	// segmask is marked motion components map. it is not used further
	cv::Mat segmask(cv::Size(curr_gray_img.cols, curr_gray_img.rows), CV_32FC1);  // Motion segmentation map.
	std::vector<cv::Rect> boundingRects;
	cv::motempl::segmentMotion(processed_mhi, segmask, boundingRects, timestamp, motion_segment_threshold);

	// FIXME [modify] >>
#if 1
	cv::Mat segmask_id;
	//csegmask.convertTo(segmask_id, CV_8SC1, 1.0, 0.0);  // Oops !!! error.
	segmask.convertTo(segmask_id, CV_8UC1, 1.0, 0.0);
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
		int idx = 0;
		for (std::vector<cv::Rect>::iterator it = boundingRects.begin(); it != boundingRects.end(); ++it, +idx)
		{
			const cv::Rect roi(*it);
			if (roi.area() < 100 || roi.width + roi.height < 100)  // reject very small components
				continue;

			const size_t count = (size_t)cv::norm(silh(roi), cv::NORM_L1);
			// check for the case of little motion
			if (count < roi.width * roi.height * 0.05)
				continue;

#if 1
			std::vector<std::vector<cv::Point> > contours;
			std::vector<cv::Vec4i> hier;
			make_contour(segmask_id, roi, idx, contours, hier);

			if (!hier.empty())
				std::transform(hier.begin(), hier.end(), std::back_inserter(hierarchy), IncreaseHierarchyOp(pointSets.size()));

			for (std::vector<std::vector<cv::Point> >::iterator itPt = contours.begin(); itPt != contours.end(); ++itPt)
				if (!itPt->empty()) pointSets.push_back(*itPt);
#else
			std::vector<cv::Point> convexHull;
			make_convex_hull(segmask_id, roi, idx, convexHull);
			if (!convexHull.empty()) pointSets.push_back(convexHull);
#endif
		}
	}
}

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
	cv::motempl::updateMotionHistory(silh, mhi, timestamp, MHI_DURATION);  // update MHI

	//
	cv::Mat processed_mhi;  // processed MHI
	{
		const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));
		const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
		const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
		const int iterations = 1;
#if 0
		cv::erode(mhi, processed_mhi, selement5, cv::Point(-1, -1), iterations);
		cv::dilate(processed_mhi, processed_mhi, selement5, cv::Point(-1, -1), iterations);
#else
		cv::morphologyEx(mhi, processed_mhi, cv::MORPH_OPEN, selement5, cv::Point(-1, -1), iterations);
		cv::morphologyEx(processed_mhi, processed_mhi, cv::MORPH_CLOSE, selement5, cv::Point(-1, -1), iterations);
#endif
	}

	// calculate motion gradient orientation and valid orientation mask
	cv::Mat mask;  // valid orientation mask
	cv::Mat orient;  // orientation
	cv::motempl::calcMotionGradient(processed_mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, motion_gradient_aperture_size);

	// segment motion: get sequence of motion components
	// segmask is marked motion components map. it is not used further
	cv::Mat segmask(cv::Size(curr_gray_img.cols, curr_gray_img.rows), CV_32FC1);  // Motion segmentation map.
	std::vector<cv::Rect> boundingRects;
	cv::motempl::segmentMotion(processed_mhi, segmask, boundingRects, timestamp, motion_segment_threshold);

	//
	//segmask.convertTo(segmentMask, CV_8SC1, 1.0, 0.0);  // Oops !!! error.
	segmask.convertTo(segmentMask, CV_8UC1, 1.0, 0.0);

	//
	double minVal = 0.0, maxVal = 0.0;
	//cv::minMaxLoc(segmask, &minVal, &maxVal);
	cv::minMaxLoc(segmentMask, &minVal, &maxVal);
	if (maxVal < 1.0e-5) return;

	// iterate through the motion components
	int idx = 0;
	for (std::vector<cv::Rect>::iterator it = boundingRects.begin(); it != boundingRects.end(); ++it, +idx)
	{
		const cv::Rect roi(*it);
		if (roi.area() < 100 || roi.width + roi.height < 100)  // reject very small components
			continue;

		if (useConvexHull)
		{
			std::vector<cv::Point> convexHull;
			make_convex_hull(segmentMask, roi, idx, convexHull);
			if (!convexHull.empty()) pointSets.push_back(convexHull);
		}
		else
		{
			std::vector<std::vector<cv::Point> > contours;
			std::vector<cv::Vec4i> hier;
			make_contour(segmentMask, roi, idx, contours, hier);

			if (!hier.empty())
				std::transform(hier.begin(), hier.end(), std::back_inserter(hierarchy), IncreaseHierarchyOp(pointSets.size()));
			std::copy(contours.begin(), contours.end(), std::back_inserter(pointSets));
		}
	}
}

void segment_motion_using_mhi(const double timestamp, const double mhiTimeDuration, const cv::Mat &prev_gray_img, const cv::Mat &curr_gray_img, cv::Mat &mhi, cv::Mat &processed_mhi, cv::Mat &component_label_map, std::vector<cv::Rect> &component_rects)
{
	cv::Mat silh;
	cv::absdiff(prev_gray_img, curr_gray_img, silh);  // get difference between frames

	const int diff_threshold = 8;
	cv::threshold(silh, silh, diff_threshold, 1.0, cv::THRESH_BINARY);  // threshold
	cv::motempl::updateMotionHistory(silh, mhi, timestamp, mhiTimeDuration);  // update MHI

	//
	{
		const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));
		const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
		const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
		const int iterations = 1;
#if 0
		cv::erode(mhi, processed_mhi, selement5, cv::Point(-1, -1), iterations);
		cv::dilate(processed_mhi, processed_mhi, selement5, cv::Point(-1, -1), iterations);
#else
		cv::morphologyEx(mhi, processed_mhi, cv::MORPH_OPEN, selement5, cv::Point(-1, -1), iterations);
		cv::morphologyEx(processed_mhi, processed_mhi, cv::MORPH_CLOSE, selement5, cv::Point(-1, -1), iterations);
#endif

#if 0
		mhi.copyTo(processed_mhi, processed_mhi);
#else
		mhi.copyTo(processed_mhi, processed_mhi > 0);
#endif
	}

	// calculate motion gradient orientation and valid orientation mask
/*
	const int motion_gradient_aperture_size = 3;
	cv::Mat motion_orientation_mask;  // valid orientation mask
	cv::Mat motion_orientation;  // orientation
	cv::calcMotionGradient(processed_mhi, motion_orientation_mask, motion_orientation, MAX_TIME_DELTA, MIN_TIME_DELTA, motion_gradient_aperture_size);
*/

	const double MAX_TIME_DELTA = 0.5;
	const double MIN_TIME_DELTA = 0.05;
	const double motion_segment_threshold = MAX_TIME_DELTA;

#if 1
	// segment motion: get sequence of motion components
	// segmask is marked motion components map. it is not used further
	cv::Mat segmask(cv::Size(curr_gray_img.cols, curr_gray_img.rows), CV_32FC1);  // Motion segmentation map.
	std::vector<cv::Rect> boundingRects;
	cv::motempl::segmentMotion(processed_mhi, segmask, boundingRects, timestamp, motion_segment_threshold);

	// iterate through the motion components
	int idx = 0;
	for (std::vector<cv::Rect>::iterator it = boundingRects.begin(); it != boundingRects.end(); ++it, +idx)
		component_rects.push_back(*it);
#else
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	const cv::Mat &tm = processed_mhi > 0;
	cv::findContours((cv::Mat &)tm, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point());

	// iterate through the motion components
	component_rects.reserve(contours.size());
	for (std::vector<std::vector<cv::Point> >::const_iterator it = contours.begin(); it != contours.end(); ++it)
		component_rects.push_back(get_bounding_box(*it));

	// FIXME [modify] >>
	component_label_map = processed_mhi > 0;
#endif
}

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

void util()
{
	if (true)
	{
#if 0
		cv::Mat bk_img(100, 100, CV_8UC1, cv::Scalar::all(0));
		cv::rectangle(bk_img, cv::Point(10, 10), cv::Point(49, 29), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		cv::rectangle(bk_img, cv::Point(60, 70), cv::Point(99, 99), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);

		// the number of pixels on outer boundary:
		//  8 connectivity	zero padding        rectangle 1					rectangle 2					total
		//	O				O					2 * (42 + 22) - 4 = 124		(41 + 31) - 1 = 71			195
		//	O				X					2 * (42 + 22) - 4 = 124		(41 + 31) - 1 = 71			195
		//	X				O					2 * (40 + 20) = 120			(40 + 30) = 70				190
		//	X				X					2 * (40 + 20) = 120			(40 + 30) = 70				190

		// the number of pixels on inner boundary:
		//  8 connectivity	zero padding        rectangle 1					rectangle 2					total
		//	O				O					2 * (40 + 20) - 4 = 116		2 * (40 + 30) - 4 = 136		252
		//	O				X					2 * (40 + 20) - 4 = 116		(40 + 30) - 1 = 69			185
		//	X				O					2 * (40 + 20) - 4 = 116		2 * (40 + 30) - 4 = 136		252
		//	X				X					2 * (40 + 20) - 4 = 116		(40 + 30) - 1 = 69			185

		const bool use_8_connectivity = true;
		const bool use_zero_padding = false;
		cv::Mat boundary_mask(bk_img.size(), CV_8UC1);
		detect_boundary(bk_img, use_8_connectivity, use_zero_padding, boundary_mask);

		const int outer_boundary_count = cv::countNonZero(1 == boundary_mask);
		const int inner_boundary_count = cv::countNonZero(2 == boundary_mask);

		std::cout << "the number of pixels on outer boundary = " << outer_boundary_count << std::endl;
		std::cout << "the number of pixels on inner boundary = " << inner_boundary_count << std::endl;

#else
		cv::Mat bk_img(100, 100, CV_8UC1, cv::Scalar::all(0));
		cv::rectangle(bk_img, cv::Point(10, 10), cv::Point(30, 30), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		cv::rectangle(bk_img, cv::Point(50, 50), cv::Point(80, 80), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		cv::rectangle(bk_img, cv::Point(60, 70), cv::Point(100, 100), CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		cv::circle(bk_img, cv::Point(70, 20), 20, CV_RGB(255, 255, 255), CV_FILLED, 8, 0);

		const bool use_8_connectivity = true;
		const bool use_zero_padding = false;
		cv::Mat boundary_mask(bk_img.size(), CV_8UC1);
		detect_boundary(bk_img, use_8_connectivity, use_zero_padding, boundary_mask);

		// display.
		cv::Mat output(bk_img.size(), CV_8UC3, cv::Scalar::all(0));
		output.setTo(cv::Scalar(0, 0, 255), 1 == boundary_mask);
		output.setTo(cv::Scalar(0, 255, 0), 2 == boundary_mask);
		output.setTo(cv::Scalar(255, 255, 255), 3 == boundary_mask);

		cv::imshow("boundary detection - input", bk_img);
		cv::imshow("boundary detection - outer", output);
		cv::imshow("boundary detection - outer boundary", 1 == boundary_mask & 0 == bk_img);
		cv::imshow("boundary detection - inner boundary", 2 == boundary_mask & bk_img > 0);
		cv::imshow("boundary detection - inner & outer boundary", (1 == boundary_mask & 0 == bk_img) | (2 == boundary_mask & bk_img > 0));
#endif
	}

	cv::waitKey(0);

	cv::destroyAllWindows();
}

}  // namespace my_opencv
