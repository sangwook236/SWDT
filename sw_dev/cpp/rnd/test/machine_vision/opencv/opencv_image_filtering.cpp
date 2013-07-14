//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <list>


namespace {
namespace local {

void mean_filtering(const cv::Mat &image, cv::Mat &filtered)
{
	// mean filter.
	const int d = -1;
	const int kernelSize = 5;
	const bool normalize = true;
	cv::boxFilter(image, filtered, d, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), normalize, cv::BORDER_REPLICATE);

	// Unnormalized box filter is useful for computing various integral characteristics over each pixel neighborhood,
	// such as covariance matrices of image derivatives (used in dense optical flow algorithms, and so on).
	// If you need to compute pixel sums over variable-size windows, use cv::integral() .
}

void bilateral_filtering(const cv::Mat &image, cv::Mat &filtered)
{
	const int d = -1;
	const double sigmaColor = 3.0;
	const double sigmaSpace = 50.0;
	cv::bilateralFilter(image, filtered, d, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);
}

void integral_image(const cv::Mat &image, cv::Mat &sum)
{
	// integral image.
	cv::Mat sqsum, tilted;
	const int sdepth = -1;
	cv::integral(image, sum, sqsum, tilted, sdepth);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void image_filtering()
{
	std::list<std::string> filenames;
#if 0
	filenames.push_back("./machine_vision_data/opencv/pic1.png");
	filenames.push_back("./machine_vision_data/opencv/pic2.png");
	filenames.push_back("./machine_vision_data/opencv/pic3.png");
	filenames.push_back("./machine_vision_data/opencv/pic4.png");
	filenames.push_back("./machine_vision_data/opencv/pic5.png");
	filenames.push_back("./machine_vision_data/opencv/pic6.png");
	filenames.push_back("./machine_vision_data/opencv/stuff.jpg");
	filenames.push_back("./machine_vision_data/opencv/synthetic_face.png");
	filenames.push_back("./machine_vision_data/opencv/puzzle.png");
	filenames.push_back("./machine_vision_data/opencv/fruits.jpg");
	filenames.push_back("./machine_vision_data/opencv/lena_rgb.bmp");
	filenames.push_back("./machine_vision_data/opencv/hand_01.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_05.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_24.jpg");
#elif 1
	filenames.push_back("./machine_vision_data/opencv/hand_left_1.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_right_1.jpg");

	filenames.push_back("./machine_vision_data/opencv/hand_01.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_02.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_03.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_04.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_05.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_06.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_07.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_08.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_09.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_10.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_11.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_12.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_13.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_14.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_15.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_16.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_17.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_18.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_19.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_20.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_21.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_22.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_23.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_24.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_25.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_26.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_27.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_28.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_29.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_30.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_31.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_32.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_33.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_34.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_35.jpg");
	filenames.push_back("./machine_vision_data/opencv/hand_36.jpg");
#elif 0
	filenames.push_back("./machine_vision_data/opencv/simple_hand_01.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_02.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_03.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_04.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_05.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_06.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_07.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_08.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_09.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_10.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_11.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_12.jpg");
	filenames.push_back("./machine_vision_data/opencv/simple_hand_13.jpg");
#endif

	cv::Mat resultant;
	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		const cv::Mat img = cv::imread(*it, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		cv::imshow("image filtering - src", img);

		// mean filtering.
		{
			{
				boost::timer::auto_cpu_timer timer;
				local::mean_filtering(img, resultant);
			}

			cv::imshow("image filtering - mean filtering", resultant);
		}

		// bilateral filtering.
		{
			{
				boost::timer::auto_cpu_timer timer;
				local::bilateral_filtering(img, resultant);
			}

			cv::imshow("image filtering - bilateral filtering", resultant);
		}

		// integral image.
		{
			{
				boost::timer::auto_cpu_timer timer;
				local::integral_image(img, resultant);  // resultant: CV_32SC3.
			}

			{
				cv::Mat img_double;
				img.convertTo(img_double, CV_64FC3, 1.0, 0.0);

				boost::timer::auto_cpu_timer timer;
				local::integral_image(img_double, resultant);  // resultant: CV_64SC3.
			}
		}

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyAllWindows();
}

}  // namespace my_opencv
