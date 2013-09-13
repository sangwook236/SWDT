#include "../lsd_lib/lsd.h"
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>


namespace {
namespace local {

void lsd_example()
{
	const int X = 128;  // x image size
	const int Y = 128;  // y image size

	// create a simple image: left half black, right half gray
	double *image = new double [X * Y];
	if (NULL == image)
	{
		std::cerr << "error: not enough memory" << std::endl;
		return;
	}
	for (int x = 0; x < X; ++x)
		for (int y = 0; y < Y; ++y)
			image[x + y * X] = (x < X / 2) ? 0.0 : 64.0;  // image(x, y)

	// LSD call
	boost::timer::cpu_timer timer;

	int n;
	double *out = lsd(&n, image, X, Y);

	boost::timer::cpu_times const elapsed_times(timer.elapsed());
	std::cout << "elpased time : " << (elapsed_times.system + elapsed_times.user) << std::endl;

	// A double array of size 7 x n_out, containing the list of line segments detected.
	// The seven values:
	//	x1, y1, x2, y2, width, p, -log10(NFA).
	//	coordinates (x1,y1) to (x2,y2), a width 'width', an angle precision of p in (0,1) given by angle_tolerance/180 degree, NFA value.
	// print output
	std::cout << n << " line segments found:" << std::endl;
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < 7; ++j)
			std::cout << out[7 * i + j] << ' ';
		std::cout << std::endl;
	}

	// free memory
	free((void *)out);
	delete [] image;
}

void lsd_image_test()
{
	std::list<std::string> filenames;
#if 1
	filenames.push_back("./data/feature_analysis/chairs.pgm");
	filenames.push_back("./data/machine_vision/opencv/hand_01_1.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_34.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_35.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_detection_ref_04_original.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_detection_ref_05_original.jpg");
#elif 0
	filenames.push_back("./data/machine_vision/opencv/hand_01.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_02.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_03.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_04.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_05.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_06.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_07.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_08.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_09.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_10.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_11.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_12.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_13.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_14.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_15.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_16.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_17.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_18.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_19.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_20.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_21.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_22.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_23.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_24.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_25.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_26.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_27.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_28.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_29.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_30.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_31.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_32.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_33.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_34.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_35.jpg");
	filenames.push_back("./data/machine_vision/opencv/hand_36.jpg");
#elif 0
	filenames.push_back("./data/machine_vision/opencv/simple_hand_01.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_02.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_03.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_04.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_05.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_06.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_07.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_08.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_09.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_10.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_11.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_12.jpg");
	filenames.push_back("./data/machine_vision/opencv/simple_hand_13.jpg");
#endif

	const std::string windowName1("LSD - gray");
	const std::string windowName2("LSD - detected");
	cv::namedWindow(windowName1);
	cv::namedWindow(windowName2);

	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
	{
		cv::Mat img = cv::imread(*it, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			return;
		}

		cv::Mat gray_img_dbl;
		{
			cv::Mat gray_img;
			cv::cvtColor(img, gray_img, CV_BGR2GRAY);

			cv::imshow(windowName1, gray_img);

#if 0
			double minVal = 0.0, maxVal = 0.0;
			cv::minMaxLoc(gray_img, &minVal, &maxVal);
			const double scale = 255.0 / (maxVal - minVal);
			const double offset = -scale * minVal;

			gray_img.convertTo(gray_img_dbl, CV_64FC1, scale, offset);
#else
			gray_img.convertTo(gray_img_dbl, CV_64FC1, 1.0, 0.0);
#endif
		}

		// call LSD
		boost::timer::cpu_timer timer;

		int numLines = 0;
		const double *lines = lsd(&numLines, (double *)gray_img_dbl.data, gray_img_dbl.cols, gray_img_dbl.rows);

		boost::timer::cpu_times const elapsed_times(timer.elapsed());
		std::cout << "elpased time : " << (elapsed_times.system + elapsed_times.user) << std::endl;

		// print output
		std::cout << numLines << " line segments found:" << std::endl;
		for (int i = 0; i < numLines; ++i)
		{
#if 0
			for (int j = 0; j < 7; ++j)
				std::cout << lines[7 * i + j] << ", ";
			std::cout << std::endl;
#endif
			// x1, y1, x2, y2, width, p, -log10(NFA)
			//cv::line(img, cv::Point(lines[7 * i + 0], lines[7 * i + 1]), cv::Point(lines[7 * i + 2], lines[7 * i + 3]), CV_RGB(255, 0, 0), lines[7 * i + 4], 8, 0);
			cv::line(img, cv::Point(lines[7 * i + 0], lines[7 * i + 1]), cv::Point(lines[7 * i + 2], lines[7 * i + 3]), CV_RGB(255, 0, 0), 2, 8, 0);
		}
		cv::imshow(windowName2, img);
		cv::waitKey(0);
	}

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_lsd {

}  // namespace my_lsd

int lsd_main(int argc, char *argv[])
{
	try
	{
		// line segment detector (LSD) --------------------
		//local::lsd_example();
		local::lsd_image_test();
	}
	catch (const cv::Exception &e)
	{
		//std::cout << "OpenCV exception caught: " << e.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception caught:" << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;

		return 1;
	}

	return 0;
}
