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
	const std::string filename("./feature_analysis_data/chairs.pgm");
	//const std::string filename("./machine_vision_data/opencv/hand_01_1.jpg");
	//const std::string filename("./machine_vision_data/opencv/hand_34.jpg");
	//const std::string filename("./machine_vision_data/opencv/hand_35.jpg");
	//const std::string filename("./machine_vision_data/opencv/hand_detection_ref_04_original.jpg");
	//const std::string filename("./machine_vision_data/opencv/hand_detection_ref_05_original.jpg");

	const std::string windowName1("LSD - gray");
	const std::string windowName2("LSD - detected");
	cv::namedWindow(windowName1);
	cv::namedWindow(windowName2);

	{
		cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << filename << std::endl;
			return;
		}

		cv::Mat gray_img;
		cv::cvtColor(img, gray_img, CV_BGR2GRAY);

		cv::imshow(windowName1, gray_img);

		cv::Mat gray_img_dbl;
		gray_img.convertTo(gray_img_dbl, CV_64FC1, 1.0, 0.0);  // TODO [check] >>

		// LSD call
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
