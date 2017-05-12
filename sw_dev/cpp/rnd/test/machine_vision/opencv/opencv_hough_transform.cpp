//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cassert>


namespace {
namespace local {

void draw_lines(const std::vector<cv::Vec2f> &lines, cv::Mat &rgb)
{
	const size_t drawingLineCount = std::min(lines.size(), (size_t)100);
	for (size_t i = 0; i < drawingLineCount; ++i)
	{
		const cv::Vec2f &line = lines[i];

		const float &rho = line[0];  // A distance between (0,0) point and the line.
		const float &theta = line[1];  // The angle between x-axis and the normal to the line.

		const double a = std::cos(theta), b = std::sin(theta);
		const double x0 = a * rho, y0 = b * rho;

		cv::Point pt1, pt2;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		cv::line(rgb, pt1, pt2, CV_RGB(255, 0, 0), 1, cv::LINE_AA, 0);
	}
}

void draw_lines(const std::vector<cv::Vec4i> &lines, cv::Mat &rgb)
{
	const size_t drawingLineCount = std::min(lines.size(), (size_t)100);
	for (size_t i = 0; i < drawingLineCount; ++i)
	{
		const cv::Vec4i &line = lines[i];

		cv::line(rgb, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), CV_RGB(255, 0, 0), 1, cv::LINE_AA, 0);
	}
}

void draw_circles(const std::vector<cv::Vec3f> &circles, cv::Mat &rgb)
{
	const size_t drawingcircleCount = std::min(circles.size(), (size_t)100);
	for (size_t i = 0; i < drawingcircleCount; ++i)
	{
		const cv::Vec3f &circle = circles[i];

		const float &x = circle[0];
		const float &y = circle[1];
		const float &r = circle[2];
		cv::circle(rgb, cv::Point(cvRound(x), cvRound(y)), 3, CV_RGB(0, 255, 0), cv::FILLED, cv::LINE_AA, 0);
		cv::circle(rgb, cv::Point(cvRound(x), cvRound(y)), cvRound(r), CV_RGB(255, 0, 0), 1, cv::LINE_AA, 0);
	}
}

void hough_transform_for_line()
{
	//const std::string img_filename("./data/machine_vision/hough_line.png");
	const std::string img_filename("./data/feature_analysis/chairs.pgm");

	// Open an image.
	cv::Mat img(cv::imread(img_filename, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cerr << "File not found: " << img_filename << std::endl;
		return;
	}

	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	// Pre-process.
	{
		//cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2.0, 2.0, cv::BORDER_DEFAULT);
		cv::Canny(gray, gray, 50.0, 200.0, 3);
	}

#if 1
	// Hough transform.
	std::vector<cv::Vec2f> lines;
	{
		const double rho = 1.0;  // Distance resolution of the accumulator in pixels.
		const double theta = CV_PI / 180.0;  // Angle resolution of the accumulator in radians.
		const int threshold = 100;  // Accumulator threshold parameter. Only those lines are returned that get enough votes.
		const double srn = 0.0;  // For the multi-scale Hough transform, it is a divisor for the distance resolution rho.
		const double stn = 0.0;  // For the multi-scale Hough transform, it is a divisor for the angle resolution theta.
		const double min_theta = 0.0;  // For standard and multi-scale Hough transform, minimum angle to check for lines. Must fall between 0 and max_theta.
		const double max_theta = CV_PI;  // For standard and multi-scale Hough transform, maximum angle to check for lines. Must fall between min_theta and CV_PI.
		cv::HoughLines(gray, lines, rho, theta, threshold, srn, stn, min_theta, max_theta);
	}

	// Draw lines.
	draw_lines(lines, img);
#else
	// Hough transform.
	std::vector<cv::Vec4i> lines;
	{
		const double rho = 1.0;  // Distance resolution of the accumulator in pixels.
		const double theta = CV_PI / 18.0;  // Angle resolution of the accumulator in radians.
		const int threshold = 80;  // Accumulator threshold parameter. Only those lines are returned that get enough votes.
		const double minLineLength = 30.0;  // Minimum line length. Line segments shorter than that are rejected.
		const double maxLineGap = 10.0;  // Maximum allowed gap between points on the same line to link them.
		cv::HoughLinesP(gray, lines, rho, theta, threshold, minLineLength, maxLineGap);
	}

	// Draw lines.
	draw_lines(lines, img);
#endif

	// Show the result.
	cv::imshow("Hough transform - Gray", gray);
	cv::imshow("Hough transform - Result", img);
	cv::waitKey();

	cv::destroyAllWindows();
}

void hough_transform_for_circle()
{
	const std::string img_filename("./data/feature_analysis/stars.pgm");

	// Open an image.
	cv::Mat img(cv::imread(img_filename, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cerr << "File not found: " << img_filename << std::endl;
		return;
	}

	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	// Pre-process.
	{
		//cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2.0, 2.0, cv::BORDER_DEFAULT);
		cv::Canny(gray, gray, 50.0, 200.0, 3);
	}

	// Hough transform.
	std::vector<cv::Vec3f> circles;
	{
		// cv::HOUGH_STANDARD, cv::HOUGH_PROBABILISTIC, cv::HOUGH_MULTI_SCALE, cv::HOUGH_GRADIENT.
		const int method = cv::HOUGH_GRADIENT;  // Currently, the only implemented method is cv::HOUGH_GRADIENT.
		const double dp = 2.0;  // Inverse ratio of the accumulator resolution to the image resolution.
		const double minDist = gray.rows / 4.0;  // Minimum distance between the centers of the detected circles.
		const double param1 = 200.0;  // First method-specific parameter. In case of cv::HOUGH_GRADIENT, it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
		const double param2 = 100.0;  // Second method-specific parameter. In case of cv::HOUGH_GRADIENT, it is the accumulator threshold for the circle centers at the detection stage.
		const int minRadius = 0;  // Minimum circle radius.
		const int maxRadius = 0;  // Maximum circle radius.
		cv::HoughCircles(gray, circles, method, dp, minDist, param1, param2, minRadius, maxRadius);
	}

	// Draw circles.
	draw_circles(circles, img);

	// Show the result.
	cv::imshow("Hough transform - Gray", gray);
	cv::imshow("Hough transform - Result", img);
	cv::waitKey();

	cv::destroyAllWindows();
}

void hough_transform_for_alignment_marker()
{
	const std::string img_filename("./data/machine_vision/alignment_marker.png");

	// Open an image.
	cv::Mat img(cv::imread(img_filename, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cerr << "File not found: " << img_filename << std::endl;
		return;
	}

	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	// Pre-process.
	{
		cv::GaussianBlur(gray, gray, cv::Size(9, 9), 9.0, 9.0, cv::BORDER_DEFAULT);

		cv::threshold(gray, gray, 100, 255, cv::THRESH_BINARY);

		const cv::Mat selement(cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1)));
		cv::morphologyEx(gray, gray, cv::MORPH_CLOSE, selement, cv::Point(-1, -1), 9);

		cv::Canny(gray, gray, 50.0, 200.0, 3);
	}

#if 1
	// Hough transform.
	std::vector<cv::Vec2f> lines;
	{
		const double rho = 1.0;  // Distance resolution of the accumulator in pixels.
		const double theta = CV_PI / 18.0;  // Angle resolution of the accumulator in radians.
		const int threshold = 40;  // Accumulator threshold parameter. Only those lines are returned that get enough votes.
		const double srn = 0.0;  // For the multi-scale Hough transform, it is a divisor for the distance resolution rho.
		const double stn = 0.0;  // For the multi-scale Hough transform, it is a divisor for the angle resolution theta.
		const double min_theta = 0.0;  // For standard and multi-scale Hough transform, minimum angle to check for lines. Must fall between 0 and max_theta.
		const double max_theta = CV_PI;  // For standard and multi-scale Hough transform, maximum angle to check for lines. Must fall between min_theta and CV_PI.
		cv::HoughLines(gray, lines, rho, theta, threshold, srn, stn, min_theta, max_theta);
	}

	// Draw lines.
	draw_lines(lines, img);
#else
	// Hough transform.
	std::vector<cv::Vec4i> lines;
	{
		const double rho = 1.0;  // Distance resolution of the accumulator in pixels.
		const double theta = CV_PI / 18.0;  // Angle resolution of the accumulator in radians.
		const int threshold = 80;  // Accumulator threshold parameter. Only those lines are returned that get enough votes.
		const double minLineLength = 30.0;  // Minimum line length. Line segments shorter than that are rejected.
		const double maxLineGap = 10.0;  // Maximum allowed gap between points on the same line to link them.
		cv::HoughLinesP(gray, lines, rho, theta, threshold, minLineLength, maxLineGap);
	}

	// Draw lines.
	draw_lines(lines, img);
#endif

	// Show the result.
	cv::imshow("Hough transform - Gray", gray);
	cv::imshow("Hough transform - Result", img);
	cv::waitKey();

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void hough_transform()
{
	//local::hough_transform_for_line();
	//local::hough_transform_for_circle();

	// Application.
	local::hough_transform_for_alignment_marker();
}

}  // namespace my_opencv
