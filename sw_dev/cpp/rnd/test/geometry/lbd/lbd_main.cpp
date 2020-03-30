#include "../lbd_lib/EDLineDetector.hh"
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <list>


namespace {
namespace local {

void edline_detector_example()
{
	std::list<std::string> img_filenames;
	img_filenames.push_back("./data/feature_analysis/chairs.pgm");
	img_filenames.push_back("./data/feature_analysis/urban_1.jpg");
	img_filenames.push_back("./data/feature_analysis/urban_2.jpg");
	img_filenames.push_back("./data/feature_analysis/urban_3.jpg");

	EDLineDetector detector;
	//std::cout << "Min. line length = " << detector.minLineLen_ << std::endl;

	for (const auto &img_filename : img_filenames)
	{
		const cv::Mat img(cv::imread(img_filename, cv::IMREAD_COLOR));
		if (img.empty())
		{
			std::cout << "Failed to load image file: " << img_filename << std::endl;
			continue;
		}

		cv::Mat gray;
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

		// Detect lines.
		{
			boost::timer::auto_cpu_timer timer;
			if (!detector.EDline(gray, false))
			{
				std::cerr << "Failed to detect lines by EDline." << std::endl;
				continue;
			}
		}

		// Show the result.
		std::cout << "\t#detected lines = " << detector.lines_.numOfLines << std::endl;
		std::cout << "\t#x-coords = " << detector.lines_.xCors.size() << std::endl;
		std::cout << "\t#y-coords = " << detector.lines_.yCors.size() << std::endl;
		std::cout << "\t#start indexes = " << detector.lines_.sId.size() << std::endl;

		cv::Mat rgb;
		img.copyTo(rgb);
		for (const auto &coords : detector.lineEndpoints_)
		{
			const float x1 = coords[0], y1 = coords[1];
			const float x2 = coords[2], y2 = coords[3];

			cv::line(rgb, cv::Point((int)std::floor(x1 + 0.5f), (int)std::floor(y1 + 0.5f)), cv::Point((int)std::floor(x2 + 0.5f), (int)std::floor(y2 + 0.5f)), CV_RGB(255, 0, 0), 1, cv::LINE_AA);
		}

		cv::imshow("LBD - Detected", rgb);
		cv::waitKey(0);
	}

	cv::destroyAllWindows();
}

void lbd_example()
{
}

}  // namespace local
}  // unnamed namespace

namespace my_lbd {

}  // namespace my_lbd

int lbd_main(int argc, char *argv[])
{
	try
	{
		// EDLine detector -----------------------------------------------------
		local::edline_detector_example();

		// Line band descriptor (LBD) & line segment matching ------------------
		//	REF [paper] >> "An efficient and robust line segment matching approach based on LBD descriptor and pairwise geometric consistency", JVCIR 2012.
		//	REF [site] >> https://github.com/mtamburrano/LBD_Descriptor
		//	REF [site] >> http://www.mip.informatik.uni-kiel.de/tiki-index.php?page=Lilian+Zhang
		//local::lbd_example();  // Not yet implemented.
	}
	catch (const cv::Exception &ex)
	{
		//std::cout << "OpenCV exception caught: " << ex.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(ex.code) << std::endl;
		std::cout << "OpenCV exception caught:" << std::endl
			<< "\tdescription: " << ex.err << std::endl
			<< "\tline:        " << ex.line << std::endl
			<< "\tfunction:    " << ex.func << std::endl
			<< "\tfile:        " << ex.file << std::endl;

		return 1;
	}

	return 0;
}
