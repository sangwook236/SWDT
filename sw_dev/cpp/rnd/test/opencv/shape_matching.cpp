#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <iostream>


namespace {

void draw_synthetic_face(cv::Mat &image, const int x, const int y)
{
	const cv::Scalar white(255);
    const cv::Scalar black(0);

	for (int j = 0; j <= 10; ++j)
	{
		const double angle = (j + 5) * CV_PI / 21;
		cv::line(image, cv::Point(cvRound(x + 100 + j*10 - 80*std::cos(angle)), cvRound(y + 100 - 90*std::sin(angle))),
			cv::Point(cvRound(x + 100 + j*10 - 30*std::cos(angle)), cvRound(y + 100 - 30*std::sin(angle))), white, 1, 8, 0);
	}

	cv::ellipse(image, cv::Point(x+150, y+100), cv::Size(100,70), 0, 0, 360, white, -1, 8, 0);
	cv::ellipse(image, cv::Point(x+115, y+70), cv::Size(30,20), 0, 0, 360, black, -1, 8, 0);
	cv::ellipse(image, cv::Point(x+185, y+70), cv::Size(30,20), 0, 0, 360, black, -1, 8, 0);
	cv::ellipse(image, cv::Point(x+115, y+70), cv::Size(15,15), 0, 0, 360, white, -1, 8, 0);
	cv::ellipse(image, cv::Point(x+185, y+70), cv::Size(15,15), 0, 0, 360, white, -1, 8, 0);
	cv::ellipse(image, cv::Point(x+115, y+70), cv::Size(5,5), 0, 0, 360, black, -1, 8, 0);
	cv::ellipse(image, cv::Point(x+185, y+70), cv::Size(5,5), 0, 0, 360, black, -1, 8, 0);
	cv::ellipse(image, cv::Point(x+150, y+100), cv::Size(10,5), 0, 0, 360, black, -1, 8, 0);
	cv::ellipse(image, cv::Point(x+150, y+150), cv::Size(40,10), 0, 0, 360, black, -1, 8, 0);
	cv::ellipse(image, cv::Point(x+27, y+100), cv::Size(20,35), 0, 0, 360, white, -1, 8, 0);
	cv::ellipse(image, cv::Point(x+273, y+100), cv::Size(20,35), 0, 0, 360, white, -1, 8, 0);
}

void generate_images(cv::Mat &image1, cv::Mat &image2)
{
#if 1
	const size_t IMAGE_WIDTH = 500;
	const size_t IMAGE_HEIGHT = 500;
	image1 = cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);

	const size_t num_rect = std::rand() % 4 + 2;
	for (size_t i = 0; i < num_rect; ++i)
	{
		const size_t width = std::rand() % 100;
		const size_t height = std::rand() % 100;
		const size_t x = std::rand() % IMAGE_WIDTH;
		const size_t y = std::rand() % IMAGE_HEIGHT;
		cv::rectangle(image1, cv::Point(x, y), cv::Point(x + width, y + height), CV_RGB(255, 255, 255), 1, 8, 0);
	}
	const size_t num_ellipse = std::rand() % 4 + 2;
	for (size_t i = 0; i < num_ellipse; ++i)
	{
		const size_t width = std::rand() % 100;
		const size_t height = std::rand() % 100;
		const size_t x = std::rand() % IMAGE_WIDTH;
		const size_t y = std::rand() % IMAGE_HEIGHT;
		const size_t angle = std::rand() % 360;
		cv::ellipse(image1, cv::Point(x, y), cv::Size(width, height), angle, 0, 360, CV_RGB(255, 255, 255), 1, 8, 0);
	}

	// warp image
	const float dx = (std::rand() % IMAGE_WIDTH) / 4.0f;
	const float dy = (std::rand() % IMAGE_HEIGHT) / 4.0f;
	const float angle = (std::rand() % 45) * (float)CV_PI / 180.0f;

	cv::Mat T(2, 3, CV_32FC1);
	T.at<float>(0,0) = (float)std::cos(angle);  T.at<float>(0,1) = (float)-std::sin(angle);  T.at<float>(0,2) = dx;
	T.at<float>(1,0) = (float)std::sin(angle);  T.at<float>(1,1) = (float)std::cos(angle);  T.at<float>(1,2) = dy;
#else
	const size_t IMAGE_WIDTH = 500;
	const size_t IMAGE_HEIGHT = 500;
	image1 = cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);

	draw_synthetic_face(image1, 0, 0);

	// warp image
	const float dx = 60.0f, dy = 20.0f;
	const float angle = 30.0f * (float)CV_PI / 180.0f;

	cv::Mat T(2, 3, CV_32FC1);
	T.at<float>(0,0) = (float)std::cos(angle);  T.at<float>(0,1) = (float)-std::sin(angle);  T.at<float>(0,2) = dx;
	T.at<float>(1,0) = (float)std::sin(angle);  T.at<float>(1,1) = (float)std::cos(angle);  T.at<float>(1,2) = dy;
#endif

	cv::warpAffine(image1, image2, T, image1.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}

void image_moment()
{
	cv::Mat image1_in, image2_in;
	generate_images(image1_in, image2_in);

	cv::Mat image1, image2;
	image1_in.convertTo(image1, CV_32FC1);
	image2_in.convertTo(image2, CV_32FC1);

	const std::string windowName1("shape matching - image1");
	const std::string windowName2("shape matching - image2");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	//
#if 1
	std::vector<std::vector<cv::Point> > contours1, contours2;
	cv::findContours(image1_in, contours1, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	cv::findContours(image2_in, contours2, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	cv::Mat img1, img2;
	cv::cvtColor(image1_in, img1, CV_GRAY2BGR);
	cv::cvtColor(image2_in, img2, CV_GRAY2BGR);

	//
	const int comparison_method = 3; //CONTOUR_MATCH_I1, CONTOUR_MATCH_I2, CONTOUR_MATCH_I3;
	size_t i = 0;
	for (std::vector<std::vector<cv::Point> >::const_iterator it1 = contours1.begin(); it1 != contours1.end(); ++it1, ++i)
	{
		double min_match_err = std::numeric_limits<double>::max();
		size_t matched_idx = -1;

		const double &startTime = (double)cv::getTickCount();

		size_t j = 0;
		for (std::vector<std::vector<cv::Point> >::const_iterator it2 = contours2.begin(); it2 != contours2.end(); ++it2, ++j)
		{
			const double match_err = cv::matchShapes(cv::Mat(*it1), cv::Mat(*it2), comparison_method, 0.0);  // use Hu moments

			if (match_err < min_match_err)
			{
				min_match_err = match_err;
				matched_idx = j;
			}
		}

		const double &endTime = (double)cv::getTickCount();
		std::cout << "\tmax match error: " << min_match_err << ", processing time: " << ((endTime - startTime) / ((double)cv::getTickFrequency() * 1000.0)) << " ms" << std::endl;

		const int r = std::rand() % 256, g = std::rand() % 256, b = std::rand() % 256;
		cv::drawContours(img1, contours1, i, CV_RGB(r, g, b), 2, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
		cv::drawContours(img2, contours2, matched_idx, CV_RGB(r, g, b), 2, 8, std::vector<cv::Vec4i>(), 0, cv::Point());

		cv::imshow(windowName1, img1);
		cv::imshow(windowName2, img2);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}
#else
	const double &startTime = (double)cv::getTickCount();

	const int comparison_method = 3; //CONTOUR_MATCH_I1, CONTOUR_MATCH_I2, CONTOUR_MATCH_I3;
	const double match_err = cv::matchShapes(image1_in, image2_in, comparison_method, 0.0);  // use Hu moments

	const double &endTime = (double)cv::getTickCount();
	std::cout << "\tmax match error: " << match_err << ", processing time: " << ((endTime - startTime) / ((double)cv::getTickFrequency() * 1000.0)) << " ms" << std::endl;
#endif

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
}

void pair_wise_geometrical_histogram()
{
	cv::Mat image1, image2;
	generate_images(image1, image2);

	const std::string windowName1("shape matching - image1");
	const std::string windowName2("shape matching - image2");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	cv::Mat img1, img2;
	cv::cvtColor(image1, img1, CV_GRAY2BGR);
	cv::cvtColor(image2, img2, CV_GRAY2BGR);

	//
	CvMemStorage *storage = cvCreateMemStorage();

	//
	CvSeq *contours1 = NULL, *contours2 = NULL;
	cvFindContours(&(IplImage)image1, storage, &contours1, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
	cvFindContours(&(IplImage)image2, storage, &contours2, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

	//
	const int dims[] = { 8, 8 };
	float range[] = { -180, 180, -100, 100 };
	float *ranges[] = { &range[0], &range[2] };

	CvHistogram *hist1 = cvCreateHist(2, (int *)dims, CV_HIST_ARRAY, ranges, 1);
	CvHistogram *hist2 = cvCreateHist(2, (int *)dims, CV_HIST_ARRAY, ranges, 1);

	CvSeq *contour1 = contours1;
	size_t i = 0;
	while (contour1)
	{
		double min_dist = std::numeric_limits<double>::max();

		const double &startTime = (double)cv::getTickCount();

		CvSeq *contour2 = contours2;
		CvSeq *matched_contour = NULL;
		while (contour2)
		{
			cvCalcPGH(contour1, hist1);
			cvCalcPGH(contour2, hist2);

			cvNormalizeHist(hist1, 100.0f);
			cvNormalizeHist(hist2, 100.0f);

			const double dist = cvCompareHist(hist1, hist2, CV_COMP_BHATTACHARYYA);
			if (dist < min_dist)
			{
				min_dist = dist;
				matched_contour = contour2;
			}

			contour2 = contour2->h_next;
		}

		const double &endTime = (double)cv::getTickCount();
		std::cout << "\tmin distance: " << min_dist << ", processing time: " << ((endTime - startTime) / ((double)cv::getTickFrequency() * 1000.0)) << " ms" << std::endl;

		const int r = std::rand() % 256, g = std::rand() % 256, b = std::rand() % 256;
		cvDrawContours(&(IplImage)img1, contour1, CV_RGB(r, g, b), CV_RGB(r, g, b), 0, 2, 8, cvPoint(0, 0));
		cvDrawContours(&(IplImage)img2, matched_contour, CV_RGB(r, g, b), CV_RGB(r, g, b), 0, 2, 8, cvPoint(0, 0));

		cv::imshow(windowName1, img1);
		cv::imshow(windowName2, img2);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;

		contour1 = contour1->h_next;
		++i;
	}

	cvReleaseHist(&hist1);
	cvReleaseHist(&hist2);

	cvReleaseMemStorage(&storage);

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
}

}

void shape_matching()
{
	// contour matching
	image_moment();
	//pair_wise_geometrical_histogram();
}
