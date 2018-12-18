//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <cassert>


namespace {
namespace local {

void outputContourInfo(const std::vector<std::vector<cv::Point> >& contours)
{
	std::cout << "#Contours = " << contours.size() << std::endl;
	size_t count = 0;
	std::cout << "#Points in each contour = ";
	for (auto contour : contours)
	{
		std::cout << contour.size() << ", ";
		count += contour.size();
	}
	std::cout << std::endl;
	std::cout << "Total number of points = " << count << std::endl;
}

void outputContourPoints(const std::vector<std::vector<cv::Point> >& contours)
{
	for (auto contour : contours)
	{
		for (auto pt : contour)
			std::cout << pt << ", ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void special_cases()
{
	const std::string windowName("Contour - Original");
	const std::string windowNameCCA("Contour - Result");

	const cv::RetrievalModes contourRetrievalModes[] = { cv::RETR_EXTERNAL, cv::RETR_LIST, cv::RETR_CCOMP, cv::RETR_TREE, cv::RETR_FLOODFILL };
	const cv::ContourApproximationModes contourApproximationModes[] = { cv::CHAIN_APPROX_NONE, cv::CHAIN_APPROX_SIMPLE, cv::CHAIN_APPROX_TC89_L1, cv::CHAIN_APPROX_TC89_KCOS };

	// NOTICE [caution] >>
	//	These contours do not contain intersection points of lines, and contain end points of lines only once.
	//	Pixels on the border might be removed.

	// Prepare a test image.
	cv::Mat gray(150, 100, CV_8UC1);
	gray.setTo(cv::Scalar::all(0));

	// T shape.
	{
		// Important pixels: *(30, 30)*, (30, 31).
		cv::line(gray, cv::Point(30, 28), cv::Point(30, 32), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(30, 30), cv::Point(32, 30), cv::Scalar::all(255), 1, cv::LINE_8);

		// Important pixels: *(50, 30)*.
		cv::line(gray, cv::Point(50, 28), cv::Point(50, 31), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(50, 30), cv::Point(51, 30), cv::Scalar::all(255), 1, cv::LINE_8);

		// Important pixels: (70, 29), (70, 30).
		cv::line(gray, cv::Point(69, 27), cv::Point(69, 29), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(69, 29), cv::Point(70, 30), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(70, 30), cv::Point(72, 30), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(69, 31), cv::Point(70, 30), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(69, 31), cv::Point(69, 33), cv::Scalar::all(255), 1, cv::LINE_8);
	}

	// Y shape.
	{
		// Important pixels: (30, 50).
		cv::line(gray, cv::Point(30, 50), cv::Point(28, 48), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(30, 50), cv::Point(32, 48), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(30, 50), cv::Point(30, 53), cv::Scalar::all(255), 1, cv::LINE_8);

		// Important pixels: (50, 50).
		cv::line(gray, cv::Point(50, 50), cv::Point(49, 49), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(50, 50), cv::Point(51, 49), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(50, 50), cv::Point(50, 53), cv::Scalar::all(255), 1, cv::LINE_8);
	}

	// Cross & diamond.
	{
		// NOTICE [info] >> Two contours below are the same.

		// Important pixels: *(30, 70)*.
		cv::line(gray, cv::Point(30, 65), cv::Point(30, 71), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(29, 70), cv::Point(31, 70), cv::Scalar::all(255), 1, cv::LINE_8);

		// Important pixels: *(50, 70)*.
		cv::line(gray, cv::Point(50, 65), cv::Point(50, 69), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(50, 69), cv::Point(49, 70), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(49, 70), cv::Point(50, 71), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(50, 71), cv::Point(51, 70), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(51, 70), cv::Point(50, 69), cv::Scalar::all(255), 1, cv::LINE_8);
	}

	// Step.
	{
		cv::line(gray, cv::Point(30, 100), cv::Point(31, 100), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(31, 100), cv::Point(31, 101), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(31, 101), cv::Point(32, 101), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(32, 101), cv::Point(32, 102), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(32, 102), cv::Point(33, 102), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(33, 102), cv::Point(33, 103), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(33, 103), cv::Point(34, 103), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(34, 103), cv::Point(34, 104), cv::Scalar::all(255), 1, cv::LINE_8);
	}

	// Adjacent vertices.
	{
		// Important pixels: *(30, 120)*, *(30, 121)*.
		cv::line(gray, cv::Point(30, 115), cv::Point(30, 125), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(30, 120), cv::Point(25, 120), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(30, 121), cv::Point(35, 121), cv::Scalar::all(255), 1, cv::LINE_8);
	}

	cv::imshow(windowName, gray);

	// Find contours.
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(gray, contours, hierarchy, contourRetrievalModes[1], contourApproximationModes[0], cv::Point(0, 0));

	// Output results.
	local::outputContourInfo(contours);

#if 0
	// Comment this out if you do not want approximation.
	std::vector<std::vector<cv::Point> > approxContours;
	for (std::vector<std::vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); ++it)
	{
		//if (it->empty()) continue;

		std::vector<cv::Point> approxCurve;
		//cv::approxPolyDP(cv::Mat(*it), approxCurve, 3.0, true);
		cv::approxPolyDP(*it, approxCurve, 3.0, true);
		approxContours.push_back(approxCurve);
	}
#endif

	{
		cv::Mat rgb;
		cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);
		cv::drawContours(rgb, contours, -1, cv::Scalar(0, 0, 255), 1, cv::LINE_8, hierarchy);

		cv::imshow(windowNameCCA, rgb);
	}

	// Display coutour points.
	local::outputContourPoints(contours);

#if 1
	// Trace contours.
	for (auto contour : contours)
		for (auto pt : contour)
		{
			cv::Mat rgb;
			cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);
			cv::circle(rgb, pt, 2, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA);

			cv::imshow(windowNameCCA + " - Contour Point", rgb);

			cv::waitKey(10);
		}
#endif
}

void line_example()
{
	const std::string windowName("Contour - Original");
	const std::string windowNameCCA("Contour - Result");

	const cv::RetrievalModes contourRetrievalModes[] = { cv::RETR_EXTERNAL, cv::RETR_LIST, cv::RETR_CCOMP, cv::RETR_TREE, cv::RETR_FLOODFILL };
	const cv::ContourApproximationModes contourApproximationModes[] = { cv::CHAIN_APPROX_NONE, cv::CHAIN_APPROX_SIMPLE, cv::CHAIN_APPROX_TC89_L1, cv::CHAIN_APPROX_TC89_KCOS };

	// NOTICE [caution] >>
	//	These contours do not contain intersection points of lines, and contain end points of lines only once.
	//	Pixels on the border might be removed.

	// Prepare a test image.
#if 1
	cv::Mat gray(200, 200, CV_8UC1);
	gray.setTo(cv::Scalar::all(0));

	// Important pixels: (100, 75), (100, 100), (100, 125), *(100, 150)* ; *(100, 199)*.
	cv::line(gray, cv::Point(100, 50), cv::Point(100, 200), cv::Scalar::all(255), 1, cv::LINE_8);
	cv::line(gray, cv::Point(100, 75), cv::Point(101, 75), cv::Scalar::all(255), 1, cv::LINE_8);
	cv::line(gray, cv::Point(98, 100), cv::Point(100, 100), cv::Scalar::all(255), 1, cv::LINE_8);
	cv::line(gray, cv::Point(100, 125), cv::Point(105, 125), cv::Scalar::all(255), 1, cv::LINE_8);
	cv::line(gray, cv::Point(95, 150), cv::Point(105, 150), cv::Scalar::all(255), 1, cv::LINE_8);
#else
	const std::string img_filepath("D:/dataset/digital_phenotyping/rda_data/20160406_trimmed_plant/adaptor1/side_0.png.thinning_cca.png");
	//const std::string img_filepath("D:/dataset/digital_phenotyping/rda_data/20160704_trimmed_plant/Set4_1/1.5xN_1(16.05.12) - ang0.png.thinning_cca.png");
	//const std::string img_filepath("D:/dataset/digital_phenotyping/rda_data/20160704_trimmed_plant/Set4_1/1.5xN_1(16.05.20) - ang0.png.thinning_cca.png");
	cv::Mat& gray = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);
#endif

	cv::imshow(windowName, gray);

	// Find contours.
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(gray, contours, hierarchy, contourRetrievalModes[1], contourApproximationModes[0], cv::Point(0, 0));

	// Output results.
	local::outputContourInfo(contours);

#if 0
	// Comment this out if you do not want approximation.
	std::vector<std::vector<cv::Point> > approxContours;
	for (std::vector<std::vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); ++it)
	{
		//if (it->empty()) continue;

		std::vector<cv::Point> approxCurve;
		//cv::approxPolyDP(cv::Mat(*it), approxCurve, 3.0, true);
		cv::approxPolyDP(*it, approxCurve, 3.0, true);
		approxContours.push_back(approxCurve);
	}
#endif

	{
		cv::Mat rgb;
		cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);
		cv::drawContours(rgb, contours, -1, cv::Scalar(0, 0, 255), 1, cv::LINE_8, hierarchy);

		std::cout << "Two pixels do not exist:" << std::endl;
		std::cout << "\tPixel (150, 100): " << rgb.at<cv::Vec3b>(150, 100) << " != [0, 0, 255]" << std::endl;
		std::cout << "\tPixel (150, 101): " << rgb.at<cv::Vec3b>(150, 101) << " == [0, 0, 255]" << std::endl;
		std::cout << "\tPixel (199, 100): " << rgb.at<cv::Vec3b>(199, 100) << " != [0, 0, 255]" << std::endl;

		cv::imshow(windowNameCCA, rgb);
	}

	// Display coutour points.
	//local::outputContourPoints(contours);

#if 1
	// Trace contours.
	for (auto contour : contours)
		for (auto pt : contour)
		{
			cv::Mat rgb;
			cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);
			cv::circle(rgb, pt, 2, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA);

			cv::imshow(windowNameCCA + " - Contour Point", rgb);

			cv::waitKey(10);
		}
#endif
}

void rectangle_example_1()
{
	const std::string windowName("Contour - Original");
	const std::string windowNameCCA("Contour - Result");

	const cv::RetrievalModes contourRetrievalModes[] = { cv::RETR_EXTERNAL, cv::RETR_LIST, cv::RETR_CCOMP, cv::RETR_TREE, cv::RETR_FLOODFILL };
	const cv::ContourApproximationModes contourApproximationModes[] = { cv::CHAIN_APPROX_NONE, cv::CHAIN_APPROX_SIMPLE, cv::CHAIN_APPROX_TC89_L1, cv::CHAIN_APPROX_TC89_KCOS };

	// NOTICE [caution] >>
	//	These contours do not contain intersection points of lines, and contain end points of lines only once.
	//	Pixels on the border might be removed.

	// Prepare a test image.
	cv::Mat gray(40, 40, CV_8UC1);
	gray.setTo(cv::Scalar::all(0));

	// Important pixels: (10, 14), (10, 16).
	cv::rectangle(gray, cv::Point(9, 10), cv::Point(10, 20), cv::Scalar::all(255), cv::FILLED, cv::LINE_8);
	cv::rectangle(gray, cv::Point(10, 14), cv::Point(15, 16), cv::Scalar::all(255), cv::FILLED, cv::LINE_8);

	cv::imshow(windowName, gray);

	// Find contours.
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(gray, contours, hierarchy, contourRetrievalModes[1], contourApproximationModes[0], cv::Point(0, 0));

	// Output results.
	local::outputContourInfo(contours);

#if 0
	// Comment this out if you do not want approximation.
	std::vector<std::vector<cv::Point> > approxContours;
	for (std::vector<std::vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); ++it)
	{
		//if (it->empty()) continue;

		std::vector<cv::Point> approxCurve;
		//cv::approxPolyDP(cv::Mat(*it), approxCurve, 3.0, true);
		cv::approxPolyDP(*it, approxCurve, 3.0, true);
		approxContours.push_back(approxCurve);
	}
#endif

	{
		cv::Mat rgb;
		cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);
		cv::drawContours(rgb, contours, -1, cv::Scalar(0, 0, 255), 1, cv::LINE_8, hierarchy);

		cv::imshow(windowNameCCA, rgb);
	}

	// Display coutour points.
	local::outputContourPoints(contours);

#if 1
	// Trace contours.
	for (auto contour : contours)
		for (auto pt : contour)
		{
			cv::Mat rgb;
			cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);
			cv::circle(rgb, pt, 2, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA);

			cv::imshow(windowNameCCA + " - Contour Point", rgb);

			cv::waitKey(10);
		}
#endif
}

void drawExternalContours(cv::Mat &img, const std::vector<std::vector<cv::Point> > &contours, const std::vector<cv::Vec4i> &hierarchy, const int idx)
{
	// For every contour of the same hierarchy level.
	for (int i = idx; i >= 0; i = hierarchy[i][0])
	{
		cv::drawContours(img, contours, i, cv::Scalar(0, 0, 255));
		//cv::rectangle(img, cv::boundingRect(contours[i]), cv::Scalar(255, 0, 0));  // Bounding box.

		// For every of its internal contours.
		for (int j = hierarchy[i][2]; j >= 0; j = hierarchy[j][0])
		{
			// Recursively print the external contours of its children.
			drawExternalContours(img, contours, hierarchy, hierarchy[j][2]);
		}
	}
}

// REF [site] >> https://stackoverflow.com/questions/19079619/efficient-way-to-combine-intersecting-bounding-rectangles
void rectangle_example_2()
{
	const std::string image_filepath("D:/work/SWDT_github/sw_dev/cpp/rnd/data/machine_vision/rectangles.png");

	const cv::Mat gray(cv::imread(image_filepath, cv::IMREAD_GRAYSCALE));
	if (gray.empty())
	{
		std::cerr << "Image file not found: " << image_filepath << std::endl;
		return;
	}

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(gray, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

	cv::Mat rgb;
	cv::cvtColor(gray, rgb, cv::COLOR_GRAY2RGB);
	drawExternalContours(rgb, contours, hierarchy, 0);

	cv::imshow("Contour - Input", gray);
	cv::imshow("Contour - Output", rgb);
	cv::waitKey(0);
}

void circle_example()
{
	const std::string windowName("Contour - Original");
	const std::string windowNameCCA("Contour - Result");

	const cv::RetrievalModes contourRetrievalModes[] = { cv::RETR_EXTERNAL, cv::RETR_LIST, cv::RETR_CCOMP, cv::RETR_TREE, cv::RETR_FLOODFILL };
	const cv::ContourApproximationModes contourApproximationModes[] = { cv::CHAIN_APPROX_NONE, cv::CHAIN_APPROX_SIMPLE, cv::CHAIN_APPROX_TC89_L1, cv::CHAIN_APPROX_TC89_KCOS };

	// Prepare a test image.
	cv::Mat gray(200, 200, CV_8UC1);
	//cv::Mat gray(200, 200, CV_32SC1);
	gray.setTo(cv::Scalar::all(0));

	cv::circle(gray, cv::Point(100, 100), 80, cv::Scalar::all(255), cv::FILLED, cv::LINE_8);
	cv::circle(gray, cv::Point(60, 100), 30, cv::Scalar::all(0), cv::FILLED, cv::LINE_8);
	cv::circle(gray, cv::Point(60, 100), 20, cv::Scalar::all(255), cv::FILLED, cv::LINE_8);
	cv::circle(gray, cv::Point(60, 100), 10, cv::Scalar::all(0), cv::FILLED, cv::LINE_8);
	cv::circle(gray, cv::Point(140, 100), 30, cv::Scalar::all(0), cv::FILLED, cv::LINE_8);

	// NOTICE [info] >> Contours are borders of object areas (white pixels), but not of blackground (black pixels).
	// Draw a line to check whether contours are contained in object areas or not
	//cv::line(gray, cv::Point(105, 21), cv::Point(101, 21), cv::Scalar::all(128), 1, cv::LINE_8);

	cv::imshow(windowName, gray);

	// Find contours.
	for (auto contourRetrievalMode : contourRetrievalModes)
	{
		const cv::ContourApproximationModes& contourApproximationMode = contourApproximationModes[0];
		//for (auto contourApproximationMode : contourApproximationModes)
		{
			std::vector<std::vector<cv::Point> > contours;
			std::vector<cv::Vec4i> hierarchy;
			cv::findContours(gray, contours, hierarchy, contourRetrievalMode, contourApproximationMode, cv::Point(0, 0));

			// Output results.
			local::outputContourInfo(contours);

#if 0
			// Comment this out if you do not want approximation.
			std::vector<std::vector<cv::Point> > approxContours;
			for (std::vector<std::vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); ++it)
			{
				//if (it->empty()) continue;

				std::vector<cv::Point> approxCurve;
				//cv::approxPolyDP(cv::Mat(*it), approxCurve, 3.0, true);
				cv::approxPolyDP(*it, approxCurve, 3.0, true);
				approxContours.push_back(approxCurve);
			}
#endif

			{
				cv::Mat rgb;
				cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);
				cv::drawContours(rgb, contours, -1, cv::Scalar(0, 0, 255), 1, cv::LINE_8, hierarchy);

				cv::imshow(windowNameCCA, rgb);
			}

			// Display coutour points.
			//local::outputContourPoints(contours);

#if 0
			// Trace contours.
			for (auto contour : contours)
				for (auto pt : contour)
				{
					cv::Mat rgb;
					cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);
					cv::circle(rgb, pt, 2, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA);

					cv::imshow(windowNameCCA + " - Contour Point", rgb);

					cv::waitKey(10);
				}
#endif

			//
			cv::waitKey(0);
		}
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void contour()
{
	local::special_cases();
	cv::waitKey(0);

	//local::line_example();
	//cv::waitKey(0);

	//local::rectangle_example_1();
	//cv::waitKey(0);

	//local::rectangle_example_2();
	//cv::waitKey(0);

	//local::circle_example();
	//cv::waitKey(0);

	cv::destroyAllWindows();
}

}  // namespace my_opencv
