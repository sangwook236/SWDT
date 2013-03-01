#include "../bgscollection_lib/bskde.h"
#include "../bgscollection_lib/sobs.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>


namespace {
namespace local {

void bskde_example()
{
	const std::string mask_filename("./change_detection_data/mask.jpg");

#if 1
	const std::string avi_filename("./change_detection_data/video.avi");
	cv::VideoCapture capture(avi_filename);
#else
	const int camId = -1;
	cv::VideoCapture capture(camId);
#endif
	if (!capture.isOpened())
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

	cv::namedWindow("bgscollection: BS_KDE - Input");
	//cv::moveWindow("bgscollection: BS_KDE - Input", 30, 0);
	cv::namedWindow("bgscollection: BS_KDE - Foreground");
	cv::namedWindow("bgscollection: BS_KDE - Mask");

	cv::Mat mask_image = cv::imread(mask_filename);
	if (!mask_filename.empty())
		cv::imshow("bgscollection: BS_KDE - Mask", mask_image);

	// create a background subtract object
	BS_KDE *bs_kde = new BS_KDE();

	cv::Mat frame, input_image, fg_mask_image;
	while ('q' != cv::waitKey(1))
	{
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}

		if (input_image.empty())
			input_image = frame.clone();
		else
			frame.copyTo(input_image);

		if (fg_mask_image.empty())
			cv::cvtColor(input_image, fg_mask_image, CV_BGR2GRAY);

		cv::imshow("bgscollection: BS_KDE - Input", input_image);

#if defined(__GNUC__)
        IplImage input_image_ipl = (IplImage)input_image, fg_mask_image_ipl = (IplImage)fg_mask_image;
		const time_t elapsedTime = bs_kde->ProcessFrame(&input_image_ipl, &fg_mask_image_ipl, 0L);
#else
		//const time_t elapsedTime = bs_kde->ProcessFrame(&(IplImage)input_image, &(IplImage)fg_mask_image, &(IplImage)mask_image);
		const time_t elapsedTime = bs_kde->ProcessFrame(&(IplImage)input_image, &(IplImage)fg_mask_image, 0L);
#endif
		if (elapsedTime)
		{
			cv::imshow("bgscollection: BS_KDE - Foreground", fg_mask_image);
			std::cout << "The bgscollection KDE time is: " << elapsedTime << std::endl;
		}
	}

	delete bs_kde;

	cv::destroyAllWindows();
}

void sobs_example()
{
#if 0
	const std::string avi_filename("./change_detection_data/video.avi");
	cv::VideoCapture capture(avi_filename);
#else
	const int camId = -1;
	cv::VideoCapture capture(camId);
#endif
	if (!capture.isOpened())
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

	SOBS detectorSOBS(50, 20, 800, 0.3, 0.1, 10, 20, 30, 40);
	//const bool retval = detectorSOBS.setROI(cv::Rect(130, 140, 500, 200));

	cv::namedWindow("bgscollection: SOBS - Input");
	cv::namedWindow("bgscollection: SOBS - Result");

	std::size_t fameIndex = 1;
	std::string filename;
	std::ostringstream sstrm;

	cv::Mat frame, input_image, fg_mask_image;
	while ('q' != cv::waitKey(1))
	{
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}

		if (input_image.empty())
			input_image = frame.clone();
		else
			frame.copyTo(input_image);

		if (fg_mask_image.empty())
			cv::cvtColor(input_image, fg_mask_image, CV_BGR2GRAY);

#if defined(__GNUC__)
        {
            IplImage input_image_ipl = (IplImage)input_image, fg_mask_image_ipl = (IplImage)fg_mask_image;
            detectorSOBS.ProcessFrame(&input_image_ipl);
            detectorSOBS.GetFgmaskImage(&fg_mask_image_ipl);
        }
#else
		detectorSOBS.ProcessFrame(&(IplImage)input_image);
		detectorSOBS.GetFgmaskImage(&(IplImage)fg_mask_image);
#endif

		sstrm << std::setfill('0') << std::setw(5) << fameIndex;

		//filename = "./change_detection_data/input" + sstrm.str() + ".bmp";
		//cv::imwrite(filename, input_image);
		//filename = "./change_detection_data/fg" + sstrm.str() + ".bmp";
		//cv::imwrite(filename, fg_mask_image);

		cv::imshow("bgscollection: SOBS - Input", input_image);
		cv::imshow("bgscollection: SOBS - Result", fg_mask_image);

		fameIndex = ++fameIndex % 100000;
	}

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_bgscollection {

}  // namespace my_bgscollection

int bgscollection_main (int argc, char *argv[])
{
	//local::bskde_example();
	local::sobs_example();

	return 0;
}
