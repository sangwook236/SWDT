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

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void connected_component()
{
	const std::string windowName("CCA - Original");
	const std::string windowNameCCA("CCA - Result");

	const cv::RetrievalModes contourRetrievalModes[] = { cv::RETR_EXTERNAL, cv::RETR_LIST, cv::RETR_CCOMP, cv::RETR_TREE, cv::RETR_FLOODFILL };
	const cv::ContourApproximationModes contourApproximationModes[] = { cv::CHAIN_APPROX_NONE, cv::CHAIN_APPROX_SIMPLE, cv::CHAIN_APPROX_TC89_L1, cv::CHAIN_APPROX_TC89_KCOS };

	{
		// Prepare a test image.
#if 1
		cv::Mat gray(200, 200, CV_8UC1);
		gray.setTo(cv::Scalar::all(0));

		cv::line(gray, cv::Point(100, 50), cv::Point(100, 175), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(100, 75), cv::Point(101, 75), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(98, 100), cv::Point(100, 100), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(100, 125), cv::Point(105, 125), cv::Scalar::all(255), 1, cv::LINE_8);
		cv::line(gray, cv::Point(95, 150), cv::Point(105, 150), cv::Scalar::all(255), 1, cv::LINE_8);
#else
		const std::string img_filepath("D:/dataset/digital_phenotyping/rda_data/20160406_trimmed_plant/adaptor1/side_0.png.thinning_cca.png");
		cv::Mat& gray = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);
#endif

		cv::imshow(windowName, gray);

		// Connected component analysis (CCA).
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(gray, contours, hierarchy, contourRetrievalModes[1], contourApproximationModes[0], cv::Point(0, 0));

		// Output results.
		{
			local::outputContourInfo(contours);

			// NOTICE [caution] >>
			//	These contours do not contain intersection points of lines,
			//	and contain end points of lines only once.
		}

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

		//
		cv::waitKey(0);
	}

	{
		// Prepare a test image.
#if 1
		cv::Mat gray(200, 200, CV_8UC1);
		//cv::Mat gray(200, 200, CV_32SC1);
		gray.setTo(cv::Scalar::all(0));

		cv::circle(gray, cv::Point(100, 100), 80, cv::Scalar::all(255), cv::FILLED, cv::LINE_8);
		cv::circle(gray, cv::Point(60, 100), 30, cv::Scalar::all(0), cv::FILLED, cv::LINE_8);
		cv::circle(gray, cv::Point(60, 100), 10, cv::Scalar::all(255), cv::FILLED, cv::LINE_8);
		cv::circle(gray, cv::Point(140, 100), 30, cv::Scalar::all(0), cv::FILLED, cv::LINE_8);
#else
		const std::string img_filepath("D:/dataset/digital_phenotyping/rda_data/20160406_trimmed_plant/adaptor1/side_0.png.thinning_cca.png");
		cv::Mat& gray = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);
#endif

		// NOTICE [info] >> Contours are borders of object areas (white pixels), but not of blackground (black pixels).
		// Draw a line to check whether contours are contained in object areas or not
		//cv::line(gray, cv::Point(105, 21), cv::Point(101, 21), cv::Scalar::all(128), 1, cv::LINE_8);

		cv::imshow(windowName, gray);

		// Connected component analysis (CCA).
		for (auto contourRetrievalMode : contourRetrievalModes)
		{
			const cv::ContourApproximationModes& contourApproximationMode = contourApproximationModes[0];
			//for (auto contourApproximationMode : contourApproximationModes)
			{
				std::vector<std::vector<cv::Point> > contours;
				std::vector<cv::Vec4i> hierarchy;
				cv::findContours(gray, contours, hierarchy, contourRetrievalMode, contourApproximationMode, cv::Point(0, 0));

				// Output results.
				{
					local::outputContourInfo(contours);
				}

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

	cv::destroyAllWindows();
}

}  // namespace my_opencv
