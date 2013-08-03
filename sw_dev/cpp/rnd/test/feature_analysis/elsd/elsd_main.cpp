#include "../elsd_lib/elsd.h"
#include "../elsd_lib/write_svg.h"
#include "../elsd_lib/valid_curve.h"
#include "../elsd_lib/process_curve.h"
#include "../elsd_lib/process_line.h"
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>


void EllipseDetection(image_double img, double rho, double prec, double p, double eps, int smooth, int *ell_count, int *circ_count, int *line_count, char *fstr);
image_double read_pgm_image_double(char * name);

namespace {
namespace local {

// ${ELSD_HOME}/elsd.c
void elsd_example()
{
	std::list<std::string> filenames;
#if 1
	filenames.push_back("./data/face_analysis/stars.pgm");
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

	const double quant = 2.0;  // bound to the quantization error on the gradient norm
	const double ang_th = 22.5;  // gradient angle tolerance in degrees
	const double p = ang_th / 180.0;
	const double prec = M_PI * ang_th / 180.0;  // radian precision
	const double rho = quant / std::sin(prec);
	const double eps = 1.0;
	const int smooth = 1;

	const std::string windowName1("ELSD - gray");
	const std::string windowName2("ELSD - detected");
	cv::namedWindow(windowName1);
	cv::namedWindow(windowName2);

	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
#if 0
		image_double img = read_pgm_image_double((char *)it->c_str());

		// call ELSD
		boost::timer::cpu_timer timer;

		int ell_count = 0, line_count = 0, circ_count = 0;
		EllipseDetection(img, rho, prec, p, eps, smooth, &ell_count, &circ_count, &line_count, (char *)it->c_str());

		boost::timer::cpu_times const elapsed_times(timer.elapsed());
		std::cout << "elpased time : " << (elapsed_times.system + elapsed_times.user) << std::endl;

		// print output
		std::cout << *it << std::endl;
		std::cout << ell_count << " elliptical arcs, " << circ_count << " circular arcs, " << line_count << " line segments" << std::endl;
#else
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

		// 두번째 file 처리 시에 오류 발생.

		image_double imgDouble = new_image_double(gray_img_dbl.cols, gray_img_dbl.rows);
		memcpy(imgDouble->data, (double *)gray_img_dbl.data, sizeof(double) * gray_img_dbl.cols * gray_img_dbl.rows);

		// call ELSD
		boost::timer::cpu_timer timer;

		int ell_count = 0, line_count = 0, circ_count = 0;
		EllipseDetection(imgDouble, rho, prec, p, eps, smooth, &ell_count, &circ_count, &line_count, (char *)it->c_str());

		boost::timer::cpu_times const elapsed_times(timer.elapsed());
		std::cout << "elpased time : " << (elapsed_times.system + elapsed_times.user) << std::endl;

		// free image_double in EllipseDetection()
		//free_image_double(imgDouble);

/*
		// print output
		std::cout << *it << std::endl;
		std::cout << ell_count << " elliptical arcs, " << circ_count << " circular arcs, " << line_count << " line segments" << std::endl;

		for (int i = 0; i < line_count; ++i)
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
*/
#endif
	}

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_elsd {

}  // namespace my_elsd

int elsd_main(int argc, char *argv[])
{
	try
	{
		// ellipse & line segment detector (ELSD) --------------------
		local::elsd_example();
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
