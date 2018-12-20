//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>
#include <vector>
#include <cmath>


namespace {
namespace local {

void contour(cv::Mat &rgb, const cv::Mat &gray)
{
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(gray, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	if (contours.empty())
	{
		std::cout << "No contour." << std::endl;
		return;
	}

    // Comment this out if you do not want approximation.
	std::vector<std::vector<cv::Point> > approxContours;
	approxContours.reserve(contours.size());
	for (std::vector<std::vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); ++it)
	{
		if (it->size() < 10)
			continue;

		std::vector<cv::Point> approxCurve;
		cv::approxPolyDP(*it, approxCurve, 3.0, true);
		approxContours.push_back(approxCurve);
	}

	//cv::drawContours(rgb, contours, -1, CV_RGB(255, 0, 0), 2, cv::LINE_AA, cv::noArray());
	cv::drawContours(rgb, approxContours, -1, CV_RGB(0, 0, 255), 2, cv::LINE_AA, cv::noArray());

	cv::imshow("Contour", rgb);
}

/*
void ellipse(IplImage *srcImg, IplImage *grayImg)
{
	const int threshold = 70;

	// create dynamic structure and sequence.
	CvMemStorage *storage = cvCreateMemStorage(0);
	CvSeq *contour = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);

	IplImage *img = cvCloneImage(grayImg);

	// threshold the source image. this needful for cvFindContours().
	cvThreshold(grayImg, img, threshold, 255, CV_THRESH_BINARY);

	// find all contours.
	cvFindContours(img, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cvPoint(0,0));

	// this cycle draw all contours and approximate it by ellipses.
	for ( ; contour; contour = contour->h_next)
	{
		const int count = contour->total;  // this is number point in contour

		// number point must be more than or equal to 6 (for cvFitEllipse_32f).
		if (count < 6) continue;

		CvMat *points_f = cvCreateMat(1, count, CV_32FC2);
		CvMat points_i = cvMat(1, count, CV_32SC2, points_f->data.ptr);
		cvCvtSeqToArray(contour, points_f->data.ptr, CV_WHOLE_SEQ);
		cvConvert(&points_i, points_f);

		// fits ellipse to current contour.
		const CvBox2D box = cvFitEllipse2(points_f);

		// draw current contour.
		cvDrawContours(srcImg, contour, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), 0, 2, 8, cvPoint(0, 0));

		// convert ellipse data from float to integer representation.
		const CvPoint center = cvPointFrom32f(box.center);
		CvSize size;
		size.width = cvRound(box.size.width * 0.5);
		size.height = cvRound(box.size.height * 0.5);

		// draw ellipse.
		cvEllipse(srcImg, center, size, -box.angle, 0, 360, CV_RGB(255, 0, 0), 2, CV_AA, 0);

		cvReleaseMat(&points_f);
	}

	cvReleaseImage(&img);
}
*/

void drawEllipseWithBox(cv::Mat &img, const cv::RotatedRect &box, const cv::Scalar &color, int lineThickness)
{
	cv::ellipse(img, box, color, lineThickness, cv::LINE_AA);

	cv::Point2f vtx[4];
	box.points(vtx);
	for (int j = 0; j < 4; ++j)
		line(img, vtx[j], vtx[(j + 1) % 4], color, lineThickness, cv::LINE_AA);
}

void drawPoints(cv::Mat &img, const std::vector<cv::Point2f> &pts, const cv::Scalar &color)
{
	for (size_t i = 0; i < pts.size(); ++i)
	{
		const cv::Point2f &pnt = pts[i];
		img.at<cv::Vec3b>(int(pnt.y), int(pnt.x))[0] = (uchar)color[0];
		img.at<cv::Vec3b>(int(pnt.y), int(pnt.x))[1] = (uchar)color[1];
		img.at<cv::Vec3b>(int(pnt.y), int(pnt.x))[2] = (uchar)color[2];
	};
}

// REF [file] >> ${OPENCV_HOME}/samples/cpp/fitellipse.cpp
void fit_ellipse(cv::Mat &rgb, const cv::Mat &gray)
{
	const int threshold = 70;
	cv::Mat bimage = gray >= threshold;

	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(bimage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	const int margin = 2;
	std::vector<std::vector<cv::Point2f> > points;
	for (size_t i = 0; i < contours.size(); ++i)
	{
		const size_t count = contours[i].size();
		if (count < 6)
			continue;

		cv::Mat pointsf;
		cv::Mat(contours[i]).convertTo(pointsf, CV_32F);

		std::vector<cv::Point2f> pts;
		for (int j = 0; j < pointsf.rows; ++j)
		{
			cv::Point2f pnt = cv::Point2f(pointsf.at<float>(j, 0), pointsf.at<float>(j, 1));
			if ((pnt.x > margin && pnt.y > margin && pnt.x < bimage.cols - margin && pnt.y < bimage.rows - margin))
			{
				if (0 == j % 20)
					pts.push_back(pnt);
			}
		}
		points.push_back(pts);
	}

	const cv::Scalar fitEllipseColor(255, 0, 0);
	const cv::Scalar fitEllipseAMSColor(0, 255, 0);
	const cv::Scalar fitEllipseDirectColor(0, 0, 255);
	const cv::Scalar fitEllipseTrueColor(255, 255, 255);

	cv::RotatedRect box, boxAMS, boxDirect;
	for (size_t i = 0; i < points.size(); ++i)
	{
		std::vector<cv::Point2f> pts = points[i];
		if (pts.size() <= 5)
			continue;

		cv::RotatedRect box = cv::fitEllipse(pts);
		if (std::max(box.size.width, box.size.height) > std::min(box.size.width, box.size.height) * 30 ||
			std::max(box.size.width, box.size.height) <= 0 ||
			std::min(box.size.width, box.size.height) <= 0)
			continue;

		cv::RotatedRect boxAMS = cv::fitEllipseAMS(pts);
		if (std::max(boxAMS.size.width, boxAMS.size.height) > std::min(boxAMS.size.width, boxAMS.size.height) * 30 ||
			std::max(box.size.width, box.size.height) <= 0 ||
			std::min(box.size.width, box.size.height) <= 0)
			continue;

		cv::RotatedRect boxDirect = cv::fitEllipseDirect(pts);
		if (std::max(boxDirect.size.width, boxDirect.size.height) > std::min(boxDirect.size.width, boxDirect.size.height) * 30 ||
			std::max(box.size.width, box.size.height) <= 0 ||
			std::min(box.size.width, box.size.height) <= 0)
			continue;

		drawEllipseWithBox(rgb, box, fitEllipseColor, 3);
		drawEllipseWithBox(rgb, boxAMS, fitEllipseAMSColor, 2);
		drawEllipseWithBox(rgb, boxDirect, fitEllipseDirectColor, 1);
		drawPoints(rgb, pts, cv::Scalar(255, 255, 255));
	}

	cv::imshow("Ellipse - Source", bimage);
	cv::imshow("Ellipse - Result", rgb);
}

/*
// helper function:
// finds a cosine of angle between vectors from pt0->pt1 and from pt0->pt2
double calc_angle(const CvPoint *pt1, const CvPoint *pt2, const CvPoint *pt0)
{
    const double dx1 = pt1->x - pt0->x;
    const double dy1 = pt1->y - pt0->y;
    const double dx2 = pt2->x - pt0->x;
    const double dy2 = pt2->y - pt0->y;
	return (dx1*dx2 + dy1*dy2) / std::sqrt((dx1*dx1 + dy1*dy1) * (dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
CvSeq * find_rectangle(IplImage *img, CvMemStorage *storage)
{
	const double threshold1 = 0.0;
	const double threshold2 = 50.0;
	const int aperture_size = 5;
	const int N = 11;

	const CvSize sz = cvSize(img->width & -2, img->height & -2);
	IplImage *timg = cvCloneImage(img);  // Make a copy of input image.
	IplImage *gray = cvCreateImage(sz, 8, 1);
	IplImage *tgray = cvCreateImage(sz, 8, 1);

	// Create empty sequence that will contain points - 4 points per square (the square's vertices).
	CvSeq *squares = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);

	// Select the maximum ROI in the image with the width and height divisible by 2.
	cvSetImageROI(timg, cvRect(0, 0, sz.width, sz.height));

	// Down-scale and up-scale the image to filter out the noise.
	IplImage *pyr = cvCreateImage(cvSize(sz.width/2, sz.height/2), 8, 3);
	cvPyrDown(timg, pyr, CV_GAUSSIAN_5x5);
	cvPyrUp(pyr, timg, CV_GAUSSIAN_5x5);
	cvReleaseImage(&pyr);

	// Find squares in every color plane of the image.
	double s, t;
	for (int c = 0; c < 3; ++c)
	{
		// Extract the c-th color plane.
		cvSetImageCOI(timg, c+1);
		cvCopy(timg, tgray, NULL);

		// Try several threshold levels.
		for (int l = 0; l < N; ++l)
		{
			// Hack: use Canny instead of zero threshold level.
			// Canny helps to catch squares with gradient shading.
			if (l == 0)
			{
				// Apply Canny. take the upper threshold from slider and set the lower to 0 (which forces edges merging).
				cvCanny(tgray, gray, threshold1, threshold2, aperture_size);
				// Dilate canny output to remove potential holes between edge segments.
				cvDilate(gray, gray, NULL, 1);
			}
			else
			{
				// Apply threshold if l != 0:
				//	tgray(x,y) = gray(x,y) < (l + 1) * 255 / N ? 255 : 0.
				cvThreshold(tgray, gray, (l+1)*255/N, 255, CV_THRESH_BINARY);
			}

			// Find contours and store them all as a list.
			CvSeq *contours = NULL;
			cvFindContours(gray, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

			// Test each contour.
			while (contours)
			{
				// Approximate contour with accuracy proportional to the contour perimeter.
				CvSeq *result = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);
				// Square contours should have 4 vertices after approximation relatively large area (to filter out noisy contours) and be convex.
				// Note: absolute value of an area is used because area may be positive or negative - in accordance with the contour orientation.
				if (result->total == 4 && cvContourArea(result, CV_WHOLE_SEQ) > 1000 && cvCheckContourConvexity(result))
				{
					s = 0;

					for (int i = 0; i < 5; ++i)
					{
						// Find minimum angle between joint edges (maximum of cosine).
						if (i >= 2)
						{
							t = std::fabs(calc_angle((CvPoint *)cvGetSeqElem(result, i), (CvPoint *)cvGetSeqElem(result, i-2), (CvPoint *)cvGetSeqElem(result, i-1)));
							s = s > t ? s : t;
						}
					}

					// If cosines of all angles are small (all angles are ~90 degree) then write quandrange vertices to resultant sequence.
					if (s < 0.3)
						for (int i = 0; i < 4; ++i)
							cvSeqPush(squares, (CvPoint *)cvGetSeqElem(result, i));
				}

				// Take the next contour.
				contours = contours->h_next;
			}
		}
	}

	cvResetImageROI(timg);

	// Release all the temporary images.
	cvReleaseImage(&gray);
	cvReleaseImage(&tgray);
	cvReleaseImage(&timg);

	return squares;
}

// The function draws all the squares in the image.
void draw_rectangle(IplImage* img, const CvSeq *squares)
{
	CvSeqReader reader;

	// Initialize reader of the sequence.
	cvStartReadSeq(squares, &reader, 0);

	// Read 4 sequence elements at a time (all vertices of a square).
	for (int i = 0; i < squares->total; i += 4)
	{
		CvPoint pt[4], *rect = pt;
		const int count = 4;

		// read 4 vertices
		CV_READ_SEQ_ELEM(pt[0], reader);
		CV_READ_SEQ_ELEM(pt[1], reader);
		CV_READ_SEQ_ELEM(pt[2], reader);
		CV_READ_SEQ_ELEM(pt[3], reader);

		// Draw the square as a closed polyline.
		cvPolyLine(img, &rect, &count, 1, 1, CV_RGB(255, 0, 0), 3, CV_AA, 0);
	}
}

void rectangle(IplImage *srcImg)
{
	CvMemStorage *storage = cvCreateMemStorage(0);

	const CvSeq *rectangles = find_rectangle(srcImg, storage);
	draw_rectangle(srcImg, rectangles);

	cvClearMemStorage(storage);
	cvReleaseMemStorage(&storage);
}
*/

double angle(const cv::Point &pt1, const cv::Point &pt2, const cv::Point &pt0)
{
	const double dx1 = pt1.x - pt0.x;
	const double dy1 = pt1.y - pt0.y;
	const double dx2 = pt2.x - pt0.x;
	const double dy2 = pt2.y - pt0.y;
	return (dx1 * dx2 + dy1 * dy2) / std::sqrt((dx1 * dx1 + dy1 * dy1)*(dx2 * dx2 + dy2 * dy2) + 1e-10);
}

// Returns sequence of squares detected on the image.
static void findSquares(const cv::Mat &image, std::vector<std::vector<cv::Point> > &squares)
{
	squares.clear();

	cv::Mat pyr, timg, gray0(image.size(), CV_8U), gray;

	// Down-scale and upscale the image to filter out the noise.
	cv::pyrDown(image, pyr, cv::Size(image.cols / 2, image.rows / 2));
	cv::pyrUp(pyr, timg, image.size());
	std::vector<std::vector<cv::Point> > contours;

	const int thresh = 50, N = 11;

	// Find squares in every color plane of the image.
	for (int c = 0; c < 3; ++c)
	{
		int ch[] = { c, 0 };
		cv::mixChannels(&timg, 1, &gray0, 1, ch, 1);

		// Try several threshold levels
		for (int l = 0; l < N; ++l)
		{
			// Hack: use Canny instead of zero threshold level.
			// Canny helps to catch squares with gradient shading.
			if (l == 0)
			{
				// Apply Canny. Take the upper threshold from slider and set the lower to 0 (which forces edges merging).
				cv::Canny(gray0, gray, 0, thresh, 5);
				// Dilate canny output to remove potential holes between edge segments.
				cv::dilate(gray, gray, cv::Mat(), cv::Point(-1, -1));
			}
			else
			{
				// Apply threshold if l != 0:
				//	tgray(x, y) = gray(x, y) < (l + 1) * 255 / N ? 255 : 0
				gray = gray0 >= (l + 1) * 255 / N;
			}

			// Find contours and store them all as a list.
			cv::findContours(gray, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

			std::vector<cv::Point> approx;
			// Test each contour.
			for (size_t i = 0; i < contours.size(); ++i)
			{
				// Approximate contour with accuracy proportional to the contour perimeter.
				cv::approxPolyDP(contours[i], approx, cv::arcLength(contours[i], true) * 0.02, true);

				// Square contours should have 4 vertices after approximation relatively large area (to filter out noisy contours) and be convex.
				// Note: absolute value of an area is used because area may be positive or negative - in accordance with the contour orientation.
				if (approx.size() == 4 && std::fabs(cv::contourArea(approx)) > 1000 && cv::isContourConvex(approx))
				{
					double maxCosine = 0;
					for (int j = 2; j < 5; ++j)
					{
						// Find the maximum cosine of the angle between joint edges.
						const double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
						maxCosine = std::max(maxCosine, cosine);
					}
					// If cosines of all angles are small (all angles are ~90 degree) then write quandrange vertices to resultant sequence.
					if (maxCosine < 0.3)
						squares.push_back(approx);
				}
			}
		}
	}
}

// The function draws all the squares in the image.
void drawSquares(cv::Mat &image, const std::vector<std::vector<cv::Point> > &squares)
{
	for (size_t i = 0; i < squares.size(); ++i)
	{
		const cv::Point *p = &squares[i][0];
		const int n = (int)squares[i].size();
		cv::polylines(image, &p, &n, 1, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
	}
}

// REF [file] >> ${OPENCV_HOME}/samples/cpp/squares.cpp
void find_rectangle(cv::Mat &rgb)
{
	std::vector<std::vector<cv::Point> > squares;
	findSquares(rgb, squares);
	drawSquares(rgb, squares);

	cv::imshow("Rectangle", rgb);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void shape_finding()
{
	std::list<std::string> filepaths;
	filepaths.push_back("../data/machine_vision/opencv/pic1.png");
	filepaths.push_back("../data/machine_vision/opencv/pic2.png");
	filepaths.push_back("../data/machine_vision/opencv/pic3.png");
	filepaths.push_back("../data/machine_vision/opencv/pic4.png");
	filepaths.push_back("../data/machine_vision/opencv/pic5.png");
	filepaths.push_back("../data/machine_vision/opencv/pic6.png");
	filepaths.push_back("../data/machine_vision/opencv/ellipses.jpg");
	filepaths.push_back("../data/machine_vision/opencv/stuff.jpg");
	filepaths.push_back("../data/machine_vision/opencv/synthetic_face.png");
	filepaths.push_back("../data/machine_vision/opencv/puzzle.png");
	filepaths.push_back("../data/machine_vision/opencv/fruits.jpg");
	filepaths.push_back("../data/machine_vision/opencv/lena_rgb.bmp");
	filepaths.push_back("../data/machine_vision/opencv/hand_01.jpg");
	filepaths.push_back("../data/machine_vision/opencv/hand_05.jpg");
	filepaths.push_back("../data/machine_vision/opencv/hand_24.jpg");

	//
	for (std::list<std::string>::iterator it = filepaths.begin(); it != filepaths.end(); ++it)
    {

		cv::Mat rgb(cv::imread(*it, cv::IMREAD_COLOR));
		if (rgb.empty())
		{
			std::cout << "File not found: " << *it << std::endl;
			continue;
		}

		cv::Mat gray;
		cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

		//local::contour(rgb, gray);
		//local::fit_ellipse(rgb, gray);
		local::find_rectangle(rgb);

		cv::waitKey(0);
	}

	cv::destroyAllWindows();
}

}  // namespace my_opencv
