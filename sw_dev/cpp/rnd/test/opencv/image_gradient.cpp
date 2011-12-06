#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <list>


namespace {

void sobel(const cv::Mat &gray, cv::Mat &xgradient, cv::Mat &ygradient, const int ksize)
{
	cv::Sobel(gray, xgradient, CV_32FC1, 1, 0, ksize, 1.0, 0.0);
	cv::Sobel(gray, ygradient, CV_32FC1, 0, 1, ksize, 1.0, 0.0);
}

}

void image_gradient()
{
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
	filenames.push_back("opencv_data\\hand_01.jpg");
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
	//filenames.push_back("opencv_data\\hand_31.jpg");
	//filenames.push_back("opencv_data\\hand_32.jpg");
#elif 0
	filenames.push_back("opencv_data\\hand_33.jpg");
	filenames.push_back("opencv_data\\hand_34.jpg");
	filenames.push_back("opencv_data\\hand_35.jpg");
	filenames.push_back("opencv_data\\hand_36.jpg");
#endif

	const std::string windowName1("image gradient - original");
	const std::string windowName2("image gradient - processed");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	//
	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		const cv::Mat img = cv::imread(*it, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		cv::Mat gray;
		if (1 == img.channels())
			img.copyTo(gray);
		else
			cv::cvtColor(img, gray, CV_BGR2GRAY);
			//cv::cvtColor(img, gray, CV_RGB2GRAY);

		//
		const int ksize = 5;
		cv::Mat xgradient, ygradient;
		//sobel(gray, xgradient, ygradient, ksize);
		sobel(gray, xgradient, ygradient, CV_SCHARR);  // use Scharr operator

		cv::Mat gradient, gradient_mask;
		cv::magnitude(xgradient, ygradient, gradient);

		const double thresholdRatio = 0.1;
		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(gradient, &minVal, &maxVal);
		cv::compare(gradient, minVal + (maxVal - minVal) * thresholdRatio, gradient_mask, cv::CMP_GT);

		cv::cvtColor(gradient_mask, gradient, CV_GRAY2BGR);

		// draw gradient
#if 0
		const float maxGradientLen = 20.0f;
		for (int r = 0; r < gradient_mask.rows; ++r)
			for (int c = 0; c < gradient_mask.cols; ++c)
			{
				const unsigned char &pix = gradient_mask.at<unsigned char>(r, c);
				if (pix)
				{
					const float &dx = xgradient.at<float>(r, c) * maxGradientLen / (float)maxVal;
					const float &dy = ygradient.at<float>(r, c) * maxGradientLen / (float)maxVal;

					cv::line(gradient, cv::Point(c, r), cv::Point(cvRound(c + dx), cvRound(r + dy)), CV_RGB(255, 0, 0), 1, 8, 0);
					//cv::circle(gradient, cv::Point(c, r), 1, CV_RGB(0, 0, 255), CV_FILLED, 8, 0);
				}
			}
#endif

		//
		cv::imshow(windowName1, img);
		cv::imshow(windowName2, gradient);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
}
