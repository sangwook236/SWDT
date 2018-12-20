//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>


namespace {
namespace local {

// REF [site] >> https://github.com/bharathp666/opencv_qr.
const int CV_QR_NORTH = 0;
const int CV_QR_EAST = 1;
const int CV_QR_SOUTH = 2;
const int CV_QR_WEST = 3;

// Function: Routine to get Distance between two points.
// Description: Given 2 points, the function returns the distance.
float cv_distance(const cv::Point2f &P, const cv::Point2f &Q)
{
	//return std::sqrt(std::pow(std::abs(P.x - Q.x), 2) + std::pow(std::abs(P.y - Q.y), 2));
	return std::sqrt(std::pow(P.x - Q.x, 2) + std::pow(P.y - Q.y, 2));
}

float cross(const cv::Point2f &v1, const cv::Point2f &v2)
{
	return v1.x*v2.y - v1.y*v2.x;
}

// Function: Perpendicular Distance of a cv::Point J from cv::line formed by Points L and M; Equation of the cv::line ax+by+c=0.
// Description: Given 3 points, the function derives the cv::line quation of the first two points,
//	  calculates and returns the perpendicular distance of the the 3rd point from this cv::line.
float cv_lineEquation(const cv::Point2f &L, const cv::Point2f &M, const cv::Point2f &J)
{
	float a, b, c, pdist;

	a = -((M.y - L.y) / (M.x - L.x));
	b = 1.0;
	c = (((M.y - L.y) / (M.x - L.x)) * L.x) - L.y;

	// Now that we have a, b, c from the equation ax + by + c, time to substitute (x,y) by values from the cv::Point J.

	pdist = (a * J.x + (b * J.y) + c) / std::sqrt((a * a) + (b * b));
	return pdist;
}

// Function: Slope of a cv::line by two Points L and M on it; Slope of cv::line, S = (x1 -x2) / (y1- y2).
// Description: Function returns the slope of the cv::line formed by given 2 points, the alignement flag
//	  indicates the cv::line is vertical and the slope is infinity.
float cv_lineSlope(const cv::Point2f &L, const cv::Point2f &M, int &alignement)
{
	float dx, dy;
	dx = M.x - L.x;
	dy = M.y - L.y;

	if (0 != dy)
	{
		alignement = 1;
		return (dy / dx);
	}
	else  // Make sure we are not dividing by zero; so use 'alignement' flag.
	{
		alignement = 0;
		return 0.0;
	}
}

// Function: Compare a point if it more far than previously recorded farthest distance.
// Description: Farthest cv::Point detection using reference point and baseline distance.
void cv_updateCorner(cv::Point2f P, cv::Point2f ref, float& baseline, cv::Point2f& corner)
{
	float temp_dist;
	temp_dist = cv_distance(P, ref);

	if (temp_dist > baseline)
	{
		baseline = temp_dist;  // The farthest distance is the new baseline.
		corner = P;  // P is now the farthest point.
	}
}

// Function: Sequence the Corners wrt to the orientation of the QR Code.
void cv_updateCornerOr(const int orientation, const std::vector<cv::Point2f> &IN, std::vector<cv::Point2f> &OUT)
{
	cv::Point2f M0, M1, M2, M3;
	if (CV_QR_NORTH == orientation)
	{
		M0 = IN[0];
		M1 = IN[1];
		M2 = IN[2];
		M3 = IN[3];
	}
	else if (CV_QR_EAST == orientation)
	{
		M0 = IN[1];
		M1 = IN[2];
		M2 = IN[3];
		M3 = IN[0];
	}
	else if (CV_QR_SOUTH == orientation)
	{
		M0 = IN[2];
		M1 = IN[3];
		M2 = IN[0];
		M3 = IN[1];
	}
	else if (CV_QR_WEST == orientation)
	{
		M0 = IN[3];
		M1 = IN[0];
		M2 = IN[1];
		M3 = IN[2];
	}

	OUT.push_back(M0);
	OUT.push_back(M1);
	OUT.push_back(M2);
	OUT.push_back(M3);
}

// Function: Routine to calculate 4 Corners of the Marker in Image Space using Region partitioning.
// Theory: OpenCV Contours stores all points that describe it and these points lie the perimeter of the polygon.
//	The below function chooses the farthest points of the polygon since they form the vertices of that polygon,
//	exactly the points we are looking for. To choose the farthest point, the polygon is divided/partitioned into
//	4 regions equal regions using bounding box. Distance algorithm is applied between the centre of bounding box
//	every contour point in that region, the farthest point is deemed as the vertex of that region. Calculating
//	for all 4 regions we obtain the 4 corners of the polygon ( - quadrilateral).
void cv_getVertices(const std::vector<std::vector<cv::Point> > &contours, int c_id, float slope, std::vector<cv::Point2f>& quad)
{
	cv::Rect box;
	box = cv::boundingRect(contours[c_id]);

	cv::Point2f M0, M1, M2, M3;
	cv::Point2f A, B, C, D, W, X, Y, Z;

	A = box.tl();
	B.x = box.br().x;
	B.y = box.tl().y;
	C = box.br();
	D.x = box.tl().x;
	D.y = box.br().y;

	W.x = (A.x + B.x) / 2;
	W.y = A.y;

	X.x = B.x;
	X.y = (B.y + C.y) / 2;

	Y.x = (C.x + D.x) / 2;
	Y.y = C.y;

	Z.x = D.x;
	Z.y = (D.y + A.y) / 2;

	float dmax[4];
	dmax[0] = 0.0;
	dmax[1] = 0.0;
	dmax[2] = 0.0;
	dmax[3] = 0.0;

	float pd1 = 0.0;
	float pd2 = 0.0;

	if (slope > 5 || slope < -5)
	{
		for (int i = 0; i < contours[c_id].size(); ++i)
		{
			pd1 = cv_lineEquation(C, A, contours[c_id][i]);  // Position of point w.r.t the diagonal AC.
			pd2 = cv_lineEquation(B, D, contours[c_id][i]);  // Position of point w.r.t the diagonal BD.

			if ((pd1 >= 0.0) && (pd2 > 0.0))
			{
				cv_updateCorner(contours[c_id][i], W, dmax[1], M1);
			}
			else if ((pd1 > 0.0) && (pd2 <= 0.0))
			{
				cv_updateCorner(contours[c_id][i], X, dmax[2], M2);
			}
			else if ((pd1 <= 0.0) && (pd2 < 0.0))
			{
				cv_updateCorner(contours[c_id][i], Y, dmax[3], M3);
			}
			else if ((pd1 < 0.0) && (pd2 >= 0.0))
			{
				cv_updateCorner(contours[c_id][i], Z, dmax[0], M0);
			}
			else
				continue;
		}
	}
	else
	{
		int halfx = (A.x + B.x) / 2;
		int halfy = (A.y + D.y) / 2;

		for (int i = 0; i < contours[c_id].size(); ++i)
		{
			if ((contours[c_id][i].x < halfx) && (contours[c_id][i].y <= halfy))
			{
				cv_updateCorner(contours[c_id][i], C, dmax[2], M0);
			}
			else if ((contours[c_id][i].x >= halfx) && (contours[c_id][i].y < halfy))
			{
				cv_updateCorner(contours[c_id][i], D, dmax[3], M1);
			}
			else if ((contours[c_id][i].x > halfx) && (contours[c_id][i].y >= halfy))
			{
				cv_updateCorner(contours[c_id][i], A, dmax[0], M2);
			}
			else if ((contours[c_id][i].x <= halfx) && (contours[c_id][i].y > halfy))
			{
				cv_updateCorner(contours[c_id][i], B, dmax[1], M3);
			}
		}
	}

	quad.push_back(M0);
	quad.push_back(M1);
	quad.push_back(M2);
	quad.push_back(M3);
}

// Function: Get the Intersection cv::Point of the lines formed by sets of two points.
bool getIntersectionPoint(const cv::Point2f &a1, const cv::Point2f &a2, const cv::Point2f &b1, const cv::Point2f &b2, cv::Point2f &intersection)
{
	cv::Point2f p = a1;
	cv::Point2f q = b1;
	cv::Point2f r(a2 - a1);
	cv::Point2f s(b2 - b1);

	if (cross(r, s) == 0) return false;

	const float t = cross(q - p, s) / cross(r, s);

	intersection = p + t*r;
	return true;
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

// REF [site] >> https://github.com/bharathp666/opencv_qr.
void qr_code()
{
	//const std::string img_filename("../data/machine_vision/qr_code_north.png");
	//const std::string img_filename("../data/machine_vision/qr_code_south.png");
	//const std::string img_filename("../data/machine_vision/qr_code_east.png");
	//const std::string img_filename("../data/machine_vision/qr_code_west.png");
	//const std::string img_filename("../data/machine_vision/qr_code_tilt.png");
	//const std::string img_filename("../data/machine_vision/qr_code_1.png");  // Failed to detect.
	//const std::string img_filename("../data/machine_vision/qr_code_2.png");  // Failed to detect.
	//const std::string img_filename("../data/machine_vision/qr_code_3.png");  // Failed to detect.
	//const std::string img_filename("../data/machine_vision/qr_code_4.png");  // Failed to detect.
	const std::string img_filename("../data/machine_vision/qr_code_5.png");  // Detect only one.
	//const std::string img_filename("../data/machine_vision/qr_code_6.png");  // Failed to detect.

	cv::Mat image = cv::imread(img_filename);
	if (image.empty())
	{
		std::cerr << "ERR: Failed to open image: " << img_filename << std::endl;
		return;
	}

	// Creation of Intermediate 'Image' Objects required later.
	cv::Mat gray(image.size(), CV_MAKETYPE(image.depth(), 1));  // To hold Grayscale Image.
	cv::Mat edges(image.size(), CV_MAKETYPE(image.depth(), 1));  // To hold Grayscale Image.
	cv::Mat traces(image.size(), CV_8UC3);  // For Debug Visuals.
	cv::Mat qr, qr_raw, qr_gray, qr_thres;

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	int mark, A, B, C, top, right, bottom, median1, median2, outlier;
	float AB, BC, CA, dist, slope, areat, arear, areab, large, padding;

	int align, orientation;

	const int DBG = 1;  // Debug flag.

	int key = 0;
	//while (key != 'q')  // While loop to query for Image Input frame.
	{

		traces = cv::Scalar(0, 0, 0);
		qr_raw = cv::Mat::zeros(100, 100, CV_8UC3);
		qr = cv::Mat::zeros(100, 100, CV_8UC3);
		qr_gray = cv::Mat::zeros(100, 100, CV_8UC1);
		qr_thres = cv::Mat::zeros(100, 100, CV_8UC1);

		//capture >> image;  // For Video input. Capture Image from Image Input.

		cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);  // Convert Image captured from Image Input to GrayScale.	
		cv::Canny(gray, edges, 100, 200, 3);  // Apply Canny edge detection on the gray image.

		cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);  // Find contours with hierarchy.

		mark = 0;  // Reset all detected marker count for this frame.

		// Get Moments for all Contours and the mass centers.
		std::vector<cv::Moments> mu(contours.size());
		std::vector<cv::Point2f> mc(contours.size());

		for (int i = 0; i < contours.size(); ++i)
		{
			mu[i] = moments(contours[i], false);
			mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
		}

		// Start processing the contour data.

		// Find Three repeatedly enclosed contours A,B,C.
		// NOTE: 1. Contour enclosing other contours is assumed to be the three Alignment markings of the QR code.
		// 2. Alternately, the Ratio of areas of the "concentric" squares can also be used for identifying base Alignment markers.
		// The below demonstrates the first method.

		for (int i = 0; i < contours.size(); ++i)
		{
			int k = i;
			int c = 0;

			while (hierarchy[k][2] != -1)
			{
				k = hierarchy[k][2];
				c = c + 1;
			}
			if (hierarchy[k][2] != -1)
				c = c + 1;

			if (c >= 5)
			{
				if (mark == 0) A = i;
				else if (mark == 1)	B = i;  // i.e., A is already found, assign current contour to B.
				else if (mark == 2)	C = i;  // i.e., A and B are already found, assign current contour to C.
				mark = mark + 1;
			}
		}

		if (mark >= 3)  // Ensure we have (at least 3; namely A,B,C) 'Alignment Markers' discovered.
		{
			// We have found the 3 markers for the QR code; Now we need to determine which of them are 'top', 'right' and 'bottom' markers.

			// Determining the 'top' marker.
			// Vertex of the triangle NOT involved in the longest side is the 'outlier'.

			AB = local::cv_distance(mc[A], mc[B]);
			BC = local::cv_distance(mc[B], mc[C]);
			CA = local::cv_distance(mc[C], mc[A]);

			if (AB > BC && AB > CA)
			{
				outlier = C; median1 = A; median2 = B;
			}
			else if (CA > AB && CA > BC)
			{
				outlier = B; median1 = A; median2 = C;
			}
			else if (BC > AB && BC > CA)
			{
				outlier = A;  median1 = B; median2 = C;
			}

			top = outlier;  // The obvious choice.

			dist = local::cv_lineEquation(mc[median1], mc[median2], mc[outlier]);  // Get the Perpendicular distance of the outlier from the longest side.			
			slope = local::cv_lineSlope(mc[median1], mc[median2], align);  // Also calculate the slope of the longest side.

			// Now that we have the orientation of the cv::line formed median1 & median2 and we also have the position of the outlier w.r.t. the cv::line.
			// Determine the 'right' and 'bottom' markers.

			if (0 == align)
			{
				bottom = median1;
				right = median2;
			}
			else if (slope < 0 && dist < 0)  // Orientation - North.
			{
				bottom = median1;
				right = median2;
				orientation = local::CV_QR_NORTH;
			}
			else if (slope > 0 && dist < 0)  // Orientation - East.
			{
				right = median1;
				bottom = median2;
				orientation = local::CV_QR_EAST;
			}
			else if (slope < 0 && dist > 0)  // Orientation - South.
			{
				right = median1;
				bottom = median2;
				orientation = local::CV_QR_SOUTH;
			}

			else if (slope > 0 && dist > 0)  // Orientation - West.
			{
				bottom = median1;
				right = median2;
				orientation = local::CV_QR_WEST;
			}

			// To ensure any unintended values do not sneak up when QR code is not present.
			float area_top, area_right, area_bottom;

			if (top < contours.size() && right < contours.size() && bottom < contours.size() && contourArea(contours[top]) > 10 && contourArea(contours[right]) > 10 && contourArea(contours[bottom]) > 10)
			{
				std::vector<cv::Point2f> L, M, O, tempL, tempM, tempO;
				cv::Point2f N;

				// src - Source Points basically the 4 end co-ordinates of the overlay image.
				// dst - Destination Points to transform overlay image.
				std::vector<cv::Point2f> src, dst;

				cv::Mat warp_matrix;

				local::cv_getVertices(contours, top, slope, tempL);
				local::cv_getVertices(contours, right, slope, tempM);
				local::cv_getVertices(contours, bottom, slope, tempO);

				local::cv_updateCornerOr(orientation, tempL, L);  // Re-arrange marker corners w.r.t orientation of the QR code.
				local::cv_updateCornerOr(orientation, tempM, M);  // Re-arrange marker corners w.r.t orientation of the QR code.
				local::cv_updateCornerOr(orientation, tempO, O);  // Re-arrange marker corners w.r.t orientation of the QR code.

				int iflag = local::getIntersectionPoint(M[1], M[2], O[3], O[2], N);

				src.push_back(L[0]);
				src.push_back(M[1]);
				src.push_back(N);
				src.push_back(O[3]);

				dst.push_back(cv::Point2f(0, 0));
				dst.push_back(cv::Point2f(qr.cols, 0));
				dst.push_back(cv::Point2f(qr.cols, qr.rows));
				dst.push_back(cv::Point2f(0, qr.rows));

				if (4 == src.size() && 4 == dst.size())  // Failsafe for WarpMatrix Calculation to have only 4 Points with src and dst.
				{
					warp_matrix = cv::getPerspectiveTransform(src, dst);
					cv::warpPerspective(image, qr_raw, warp_matrix, cv::Size(qr.cols, qr.rows));
					cv::copyMakeBorder(qr_raw, qr, 10, 10, 10, 10, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

					cv::cvtColor(qr, qr_gray, cv::COLOR_RGB2GRAY);
					cv::threshold(qr_gray, qr_thres, 127, 255, cv::THRESH_BINARY);

					//cv::threshold(qr_gray, qr_thres, 0, 255, cv::THRESH_OTSU);
					//for (int d = 0 ; d < 4; ++d)  {  src.pop_back(); dst.pop_back();  }
				}

				// Draw contours on the image.
				cv::drawContours(image, contours, top, cv::Scalar(255, 200, 0), 2, 8, hierarchy, 0);
				cv::drawContours(image, contours, right, cv::Scalar(0, 0, 255), 2, 8, hierarchy, 0);
				cv::drawContours(image, contours, bottom, cv::Scalar(255, 0, 100), 2, 8, hierarchy, 0);

				// Insert Debug instructions here.
				if (DBG == 1)
				{
					// Debug Prints.
					// Visualizations for ease of understanding.
					if (slope > 5)
						cv::circle(traces, cv::Point(10, 20), 5, cv::Scalar(0, 0, 255), -1, 8, 0);
					else if (slope < -5)
						cv::circle(traces, cv::Point(10, 20), 5, cv::Scalar(255, 255, 255), -1, 8, 0);

					// Draw contours on Trace image for analysis.
					cv::drawContours(traces, contours, top, cv::Scalar(255, 0, 100), 1, 8, hierarchy, 0);
					cv::drawContours(traces, contours, right, cv::Scalar(255, 0, 100), 1, 8, hierarchy, 0);
					cv::drawContours(traces, contours, bottom, cv::Scalar(255, 0, 100), 1, 8, hierarchy, 0);

					// Draw points (4 corners) on Trace image for each Identification marker.
					cv::circle(traces, L[0], 2, cv::Scalar(255, 255, 0), -1, 8, 0);
					cv::circle(traces, L[1], 2, cv::Scalar(0, 255, 0), -1, 8, 0);
					cv::circle(traces, L[2], 2, cv::Scalar(0, 0, 255), -1, 8, 0);
					cv::circle(traces, L[3], 2, cv::Scalar(128, 128, 128), -1, 8, 0);

					cv::circle(traces, M[0], 2, cv::Scalar(255, 255, 0), -1, 8, 0);
					cv::circle(traces, M[1], 2, cv::Scalar(0, 255, 0), -1, 8, 0);
					cv::circle(traces, M[2], 2, cv::Scalar(0, 0, 255), -1, 8, 0);
					cv::circle(traces, M[3], 2, cv::Scalar(128, 128, 128), -1, 8, 0);

					cv::circle(traces, O[0], 2, cv::Scalar(255, 255, 0), -1, 8, 0);
					cv::circle(traces, O[1], 2, cv::Scalar(0, 255, 0), -1, 8, 0);
					cv::circle(traces, O[2], 2, cv::Scalar(0, 0, 255), -1, 8, 0);
					cv::circle(traces, O[3], 2, cv::Scalar(128, 128, 128), -1, 8, 0);

					// Draw point of the estimated 4th Corner of (entire) QR Code.
					cv::circle(traces, N, 2, cv::Scalar(255, 255, 255), -1, 8, 0);

					// Draw the lines used for estimating the 4th Corner of QR Code.
					cv::line(traces, M[1], N, cv::Scalar(0, 0, 255), 1, 8, 0);
					cv::line(traces, O[3], N, cv::Scalar(0, 0, 255), 1, 8, 0);

					// Show the Orientation of the QR Code wrt to 2D Image Space.
					int fontFace = cv::FONT_HERSHEY_PLAIN;

					if (local::CV_QR_NORTH == orientation)
					{
						cv::putText(traces, "NORTH", cv::Point(20, 30), fontFace, 1, cv::Scalar(0, 255, 0), 1, 8);
					}
					else if (local::CV_QR_EAST == orientation)
					{
						cv::putText(traces, "EAST", cv::Point(20, 30), fontFace, 1, cv::Scalar(0, 255, 0), 1, 8);
					}
					else if (local::CV_QR_SOUTH == orientation)
					{
						cv::putText(traces, "SOUTH", cv::Point(20, 30), fontFace, 1, cv::Scalar(0, 255, 0), 1, 8);
					}
					else if (local::CV_QR_WEST == orientation)
					{
						cv::putText(traces, "WEST", cv::Point(20, 30), fontFace, 1, cv::Scalar(0, 255, 0), 1, 8);
					}

					// Debug Prints
				}

			}
		}

		cv::imshow("Image", image);
		cv::imshow("Traces", traces);
		cv::imshow("QR code", qr_thres);

		key = cv::waitKey();  // OPENCV: wait for 1ms before accessing next frame.
	}

	cv::destroyAllWindows();
}

}  // namespace my_opencv
