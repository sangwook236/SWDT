//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
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
	//const std::string img_filename("../data/machine_vision/building.jpg");
	const std::string img_filename("../data/feature_analysis/chairs.pgm");

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

// REF [site] >> https://docs.opencv.org/4.1.2/dd/d1a/group__imgproc__feature.html
void hough_transform_for_line_using_point_set()
{
	const static float Points[20][2] = {
		{ 0.0f,   369.0f }, { 10.0f,  364.0f }, { 20.0f,  358.0f }, { 30.0f,  352.0f },
		{ 40.0f,  346.0f }, { 50.0f,  341.0f }, { 60.0f,  335.0f }, { 70.0f,  329.0f },
		{ 80.0f,  323.0f }, { 90.0f,  318.0f }, { 100.0f, 312.0f }, { 110.0f, 306.0f },
		{ 120.0f, 300.0f }, { 130.0f, 295.0f }, { 140.0f, 289.0f }, { 150.0f, 284.0f },
		{ 160.0f, 277.0f }, { 170.0f, 271.0f }, { 180.0f, 266.0f }, { 190.0f, 260.0f }
	};

	std::vector<cv::Point2f> point;
	for (int i = 0; i < 20; ++i)
		point.push_back(cv::Point2f(Points[i][0], Points[i][1]));

	const double rhoMin = 0.0f, rhoMax = 360.0f, rhoStep = 1.0;
	const double thetaMin = 0.0, thetaMax = CV_PI / 2.0, thetaStep = CV_PI / 180.0;
	cv::Mat lines;
	cv::HoughLinesPointSet(point, lines, 20, 1, rhoMin, rhoMax, rhoStep, thetaMin, thetaMax, thetaStep);

	std::vector<cv::Vec3d> line3d;
	lines.copyTo(line3d);
	for (int i = 0; i < line3d.size(); ++i)
		std::cout << "#" << i << ": votes = " << (int)line3d[i].val[0] << ", rho = " << line3d[i].val[1] << ", theta = " << line3d[i].val[2] << std::endl;
}

bool getEdges(const cv::Mat &src, cv::Mat &dst)
{
	cv::Mat ucharSingleSrc;
	src.convertTo(ucharSingleSrc, CV_8UC1);

	cv::Canny(ucharSingleSrc, dst, 50, 200, 3);
	return true;
}

bool fht(const cv::Mat &src, cv::Mat &dst, int dstDepth, int angleRange, int op, int skew)
{
	clock_t clocks = std::clock();

	cv::ximgproc::FastHoughTransform(src, dst, dstDepth, angleRange, op, skew);

	clocks = std::clock() - clocks;
	double secs = (double)clocks / CLOCKS_PER_SEC;
	std::cout << std::setprecision(2) << "FastHoughTransform finished in " << secs << " seconds" << std::endl;

	return true;
}

template<typename T>
bool rel(std::pair<T, cv::Point> const &a, std::pair<T, cv::Point> const &b)
{
	return a.first > b.first;
}

template<typename T>
bool incIfGreater(const T& a, const T& b, int *value)
{
	if (!value || a < b)
		return false;
	if (a > b)
		++(*value);
	return true;
}

template<typename T>
bool getLocalExtr(std::vector<cv::Vec4i> &lines, const cv::Mat &src, const cv::Mat &fht, float minWeight, int maxCount)
{
	const int MAX_LEN = 10000;

	std::vector<std::pair<T, cv::Point> > weightedPoints;
	for (int y = 0; y < fht.rows; ++y)
	{
		if (weightedPoints.size() > MAX_LEN)
			break;

		T const *pLine = (T *)fht.ptr(std::max(y - 1, 0));
		T const *cLine = (T *)fht.ptr(y);
		T const *nLine = (T *)fht.ptr(std::min(y + 1, fht.rows - 1));

		for (int x = 0; x < fht.cols; ++x)
		{
			if (weightedPoints.size() > MAX_LEN)
				break;

			T const value = cLine[x];
			if (value >= minWeight)
			{
				int isLocalMax = 0;
				for (int xx = std::max(x - 1, 0); xx <= std::min(x + 1, fht.cols - 1); ++xx)
				{
					if (!incIfGreater(value, pLine[xx], &isLocalMax) ||
						!incIfGreater(value, cLine[xx], &isLocalMax) ||
						!incIfGreater(value, nLine[xx], &isLocalMax))
					{
						isLocalMax = 0;
						break;
					}
				}
				if (isLocalMax > 0)
					weightedPoints.push_back(std::make_pair(value, cv::Point(x, y)));
			}
		}
	}

	if (weightedPoints.empty())
		return true;

	std::sort(weightedPoints.begin(), weightedPoints.end(), &rel<T>);
	weightedPoints.resize(std::min(static_cast<int>(weightedPoints.size()), maxCount));

	for (size_t i = 0; i < weightedPoints.size(); ++i)
	{
		lines.push_back(cv::ximgproc::HoughPoint2Line(weightedPoints[i].second, src));
	}
	return true;
}

bool getLocalExtr(std::vector<cv::Vec4i> &lines, const cv::Mat &src, const cv::Mat &fht, float minWeight, int maxCount)
{
	int const depth = CV_MAT_DEPTH(fht.type());
	switch (depth)
	{
	case 0:
		return getLocalExtr<uchar>(lines, src, fht, minWeight, maxCount);
	case 1:
		return getLocalExtr<schar>(lines, src, fht, minWeight, maxCount);
	case 2:
		return getLocalExtr<ushort>(lines, src, fht, minWeight, maxCount);
	case 3:
		return getLocalExtr<short>(lines, src, fht, minWeight, maxCount);
	case 4:
		return getLocalExtr<int>(lines, src, fht, minWeight, maxCount);
	case 5:
		return getLocalExtr<float>(lines, src, fht, minWeight, maxCount);
	case 6:
		return getLocalExtr<double>(lines, src, fht, minWeight, maxCount);
	default:
		return false;
	}
}

void rescale(cv::Mat const &src, cv::Mat &dst, int const maxHeight = 500, int const maxWidth = 1000)
{
	double scale = std::min(std::min(static_cast<double>(maxWidth) / src.cols, static_cast<double>(maxHeight) / src.rows), 1.0);
	cv::resize(src, dst, cv::Size(), scale, scale, cv::INTER_LINEAR_EXACT);
}

void showHumanReadableImg(std::string const &name, cv::Mat const &img)
{
	cv::Mat ucharImg;
	img.convertTo(ucharImg, CV_MAKETYPE(CV_8U, img.channels()));
	rescale(ucharImg, ucharImg);
	cv::imshow(name, ucharImg);
}

void showFht(cv::Mat const &fht)
{
	double minv(0), maxv(0);
	cv::minMaxLoc(fht, &minv, &maxv);
	cv::Mat ucharFht;
	fht.convertTo(ucharFht, CV_MAKETYPE(CV_8U, fht.channels()), 255.0 / (maxv + minv), minv / (maxv + minv));
	rescale(ucharFht, ucharFht);
	cv::imshow("Fast hough transform", ucharFht);
}

void showLines(cv::Mat const &src, std::vector<cv::Vec4i> const &lines)
{
	cv::Mat bgrSrc;
	cv::cvtColor(src, bgrSrc, cv::COLOR_GRAY2BGR);

	for (size_t i = 0; i < lines.size(); ++i)
	{
		cv::Vec4i const &l = lines[i];
		cv::line(bgrSrc, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
	}

	rescale(bgrSrc, bgrSrc);
	cv::imshow("lines", bgrSrc);
}

// REF [file] >> ${OPENCV_CONTRIB_HOME}/modules/ximgproc/samples/fast_hough_transform.cpp
void fast_hough_transform()
{
	const std::string img_filename("../data/machine_vision/building.jpg");

	// Open an image.
	cv::Mat src(cv::imread(img_filename, cv::IMREAD_GRAYSCALE));
	if (src.empty())
	{
		std::cerr << "File not found: " << img_filename << std::endl;
		return;
	}

	cv::pyrDown(src, src);  cv::pyrUp(src, src);
	//cv::GaussianBlur(src, src, cv::Size(5, 5), 0);
	//cv::medianBlur(src, src, 5);
	//const double clipLimit = 40.0;
	//const cv::Size tileGridSize(8, 8);
	//cv::Ptr<cv::CLAHE> clahe(cv::createCLAHE(clipLimit, tileGridSize));
	//clahe->apply(src, src);

	const int depth = CV_32S;
	const int angleRange = cv::ximgproc::ARO_315_135;
	const int op = cv::ximgproc::FHT_ADD;
	//const int op = cv::ximgproc::FHT_AVE;
	const int skew = cv::ximgproc::HDO_DESKEW;

	showHumanReadableImg("Image", src);

	cv::Mat canny;
	if (!getEdges(src, canny))
	{
		std::cout << "Failed to select canny edges";
		return;
	}
	showHumanReadableImg("canny", canny);

	cv::Mat hough;
	if (!fht(canny, hough, depth, angleRange, op, skew))
	{
		std::cout << "Failed to compute Fast Hough Transform";
		return;
	}
	showFht(hough);

	std::vector<cv::Vec4i> lines;
	if (!getLocalExtr(lines, canny, hough, static_cast<float>(255 * 0.3 * std::min(src.rows, src.cols)), 50))
	{
		std::cout << "Failed to find local maximums on FHT image";
		return;
	}
	showLines(canny, lines);

	cv::waitKey(0);
	cv::destroyAllWindows();
}

// REF [paper] >> 
//	"Accurate and robust line segment extraction by analyzing distribution around peaks in Hough space", CVIU 2003.
//	"Rectangle Detection based on a Windowed Hough Transform", SIBGRAPI 2004.
void enhance_hough_space(const cv::Mat &hough_space, cv::Mat &enhanced_hough_space, const size_t kernel_width, const size_t kernel_height)
{
	hough_space.convertTo(enhanced_hough_space, CV_32FC1);

	cv::Mat hough_integral;
	cv::boxFilter(enhanced_hough_space, hough_integral, -1, cv::Size(kernel_width, kernel_height), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
	cv::pow(enhanced_hough_space, 2, enhanced_hough_space);
	cv::divide(enhanced_hough_space, hough_integral, enhanced_hough_space);
}

// REF [site] >> http://fourier.eng.hmc.edu/e161/dipum/houghpeaks.m
std::list<cv::Point> find_peaks_in_hough_space(const cv::Mat &hough, const cv::Size &neighborhood, const size_t num_peaks, const float threshold)
{
	cv::Mat hough_new;
	hough.copyTo(hough_new);

	const int half_width = neighborhood.width / 2, half_height = neighborhood.height / 2;
	std::list<cv::Point> peaks;
	cv::Point maxPt;
	double maxVal;
	while (true)
	{
		cv::minMaxLoc(hough_new, nullptr, &maxVal, nullptr, &maxPt);
		if (maxVal < threshold) break;

		peaks.push_back(maxPt);
		if (peaks.size() >= num_peaks) break;

		const cv::Rect rct(cv::Point(std::max(maxPt.x - half_width, 0), std::max(maxPt.y - half_height, 0)), cv::Point(std::min(maxPt.x + half_width, hough_new.cols - 1), std::min(maxPt.y + half_height, hough_new.rows - 1)));
		hough_new(rct).setTo(cv::Scalar::all(0));
	}

	return peaks;
}

// REF [site] >> http://amroamroamro.github.io/mexopencv/opencv_contrib/fast_hough_transform_demo.html
bool hough_space_analysis()
{
	const std::string img_filename("../data/machine_vision/building.jpg");

	// Open an image.
	cv::Mat img(cv::imread(img_filename, cv::IMREAD_GRAYSCALE));
	if (img.empty())
	{
		std::cerr << "File not found: " << img_filename << std::endl;
		return;
	}

	cv::pyrDown(img, img);  cv::pyrUp(img, img);
	//cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
	//cv::medianBlur(img, img, 5);
	//const double clipLimit = 40.0;
	//const cv::Size tileGridSize(8, 8);
	//cv::Ptr<cv::CLAHE> clahe(cv::createCLAHE(clipLimit, tileGridSize));
	//clahe->apply(img, img);

	cv::Mat edge;
	cv::Canny(img, edge, 50, 200, 3);

	cv::imshow("Canny", edge);

	cv::Mat hough_space;
	//cv::ximgproc::FastHoughTransform(edge, hough_space, CV_32S, cv::ximgproc::ARO_315_135, cv::ximgproc::FHT_ADD, cv::ximgproc::HDO_DESKEW);
	cv::ximgproc::FastHoughTransform(edge, hough_space, CV_32F, cv::ximgproc::ARO_315_135, cv::ximgproc::FHT_AVE, cv::ximgproc::HDO_DESKEW);

	{
		double minVal, maxVal;
		cv::minMaxLoc(hough_space, &minVal, &maxVal);

		cv::imshow("Hough Space", hough_space / (float)maxVal);
		//cv::imwrite("hough_space.tif", hough_space / (float)maxVal);
	}

#if true
	//--------------------
	enhance_hough_space(hough_space, hough_space, 10, 10);

	{
		double minVal, maxVal;
		cv::minMaxLoc(hough_space, &minVal, &maxVal);

		cv::imshow("Enhanced Hough Space", hough_space / (float)maxVal);
		//cv::imwrite("enhanced_hough_space.tif", hough_space / (float)maxVal);
	}
#endif

	//--------------------
	const cv::Size neighborhood(50, 50);
	const size_t num_peaks = 10;
	const float threshold = 50.0f;
	const std::list<cv::Point> &hough_peaks = find_peaks_in_hough_space(hough_space, neighborhood, num_peaks, threshold);

	{
		double maxVal;
		cv::minMaxLoc(hough_space, nullptr, &maxVal);

		cv::Mat hough_rgb;
		hough_space.convertTo(hough_rgb, CV_8U, 255.0 / maxVal, 0.0);
		cv::cvtColor(hough_rgb, hough_rgb, cv::COLOR_GRAY2BGR);
		for (const auto &peak : hough_peaks)
			cv::circle(hough_rgb, peak, 3, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0);
		cv::imshow("Hough Space with Peaks", hough_rgb);
		//cv::imwrite("hough_space_with_peaks.tif", hough_rgb);

		cv::Mat line_rgb;
		//cv::cvtColor(edge, line_rgb, cv::COLOR_GRAY2BGR);
		cv::cvtColor(img, line_rgb, cv::COLOR_GRAY2BGR);
		for (const auto &peak : hough_peaks)
		{
			const cv::Vec4i &line = cv::ximgproc::HoughPoint2Line(peak, edge, cv::ximgproc::ARO_315_135, cv::ximgproc::HDO_DESKEW, cv::ximgproc::RO_IGNORE_BORDERS);
			cv::line(line_rgb, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0);
		}
		cv::imshow("Hough Lines", line_rgb);
	}

	return true;
}

void generalized_hough_transform()
{
	cv::Ptr<cv::GeneralizedHoughBallard> ght = cv::createGeneralizedHoughBallard();
	//cv::Ptr<cv::GeneralizedHoughGuil> ght = cv::createGeneralizedHoughGuil();

	//ght->setLevels(levels);  // R-table levels.
	//ght->setVotesThreshold(votesThreshold);  // The accumulator threshold for the template centers at the detection stage.

	std::cout << "Levels = " << ght->getLevels() << std::endl;
	std::cout << "Votes threshol = " << ght->getVotesThreshold() << std::endl;
}

void hough_transform_for_alignment_marker()
{
	const std::string img_filename("../data/machine_vision/alignment_marker.png");

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
	//local::hough_transform_for_line_using_point_set();
	local::fast_hough_transform();
	//local::hough_space_analysis();

	// Generalized Hough transform.
	//local::generalized_hough_transform();  // Not yet implemented.

	// Application.
	//local::hough_transform_for_alignment_marker();
}

}  // namespace my_opencv
