#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <list>


namespace {

void bilateral_filtering(const cv::Mat &image, cv::Mat &filtered)
{
	const int d = -1;
	const double sigmaColor = 3.0;
	const double sigmaSpace = 50.0;
	cv::bilateralFilter(image, filtered, d, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);
}

}

void image_filtering()
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
	filenames.push_back("opencv_data\\hand_left_1.jpg");
	filenames.push_back("opencv_data\\hand_right_1.jpg");

	filenames.push_back("opencv_data\\hand_01.jpg");
	filenames.push_back("opencv_data\\hand_02.jpg");
	filenames.push_back("opencv_data\\hand_03.jpg");
	filenames.push_back("opencv_data\\hand_04.jpg");
	filenames.push_back("opencv_data\\hand_05.jpg");
	filenames.push_back("opencv_data\\hand_06.jpg");
	filenames.push_back("opencv_data\\hand_07.jpg");
	filenames.push_back("opencv_data\\hand_08.jpg");
	filenames.push_back("opencv_data\\hand_09.jpg");
	filenames.push_back("opencv_data\\hand_10.jpg");
	filenames.push_back("opencv_data\\hand_11.jpg");
	filenames.push_back("opencv_data\\hand_12.jpg");
	filenames.push_back("opencv_data\\hand_13.jpg");
	filenames.push_back("opencv_data\\hand_14.jpg");
	filenames.push_back("opencv_data\\hand_15.jpg");
	filenames.push_back("opencv_data\\hand_16.jpg");
	filenames.push_back("opencv_data\\hand_17.jpg");
	filenames.push_back("opencv_data\\hand_18.jpg");
	filenames.push_back("opencv_data\\hand_19.jpg");
	filenames.push_back("opencv_data\\hand_20.jpg");
	filenames.push_back("opencv_data\\hand_21.jpg");
	filenames.push_back("opencv_data\\hand_22.jpg");
	filenames.push_back("opencv_data\\hand_23.jpg");
	filenames.push_back("opencv_data\\hand_24.jpg");
	filenames.push_back("opencv_data\\hand_25.jpg");
	filenames.push_back("opencv_data\\hand_26.jpg");
	filenames.push_back("opencv_data\\hand_27.jpg");
	filenames.push_back("opencv_data\\hand_28.jpg");
	filenames.push_back("opencv_data\\hand_29.jpg");
	filenames.push_back("opencv_data\\hand_30.jpg");
	filenames.push_back("opencv_data\\hand_31.jpg");
	filenames.push_back("opencv_data\\hand_32.jpg");
	filenames.push_back("opencv_data\\hand_33.jpg");
	filenames.push_back("opencv_data\\hand_34.jpg");
	filenames.push_back("opencv_data\\hand_35.jpg");
	filenames.push_back("opencv_data\\hand_36.jpg");
#elif 0
	filenames.push_back("opencv_data\\simple_hand_01.jpg");
	filenames.push_back("opencv_data\\simple_hand_02.jpg");
	filenames.push_back("opencv_data\\simple_hand_03.jpg");
	filenames.push_back("opencv_data\\simple_hand_04.jpg");
	filenames.push_back("opencv_data\\simple_hand_05.jpg");
	filenames.push_back("opencv_data\\simple_hand_06.jpg");
	filenames.push_back("opencv_data\\simple_hand_07.jpg");
	filenames.push_back("opencv_data\\simple_hand_08.jpg");
	filenames.push_back("opencv_data\\simple_hand_09.jpg");
	filenames.push_back("opencv_data\\simple_hand_10.jpg");
	filenames.push_back("opencv_data\\simple_hand_11.jpg");
	filenames.push_back("opencv_data\\simple_hand_12.jpg");
	filenames.push_back("opencv_data\\simple_hand_13.jpg");
#endif

	const std::string windowName1("image filtering - original");
	const std::string windowName2("image filtering - filtered");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		const cv::Mat img = cv::imread(*it, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		cv::Mat filtered;
		bilateral_filtering(img, filtered);

		cv::imshow(windowName1, img);
		cv::imshow(windowName2, filtered);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
}
