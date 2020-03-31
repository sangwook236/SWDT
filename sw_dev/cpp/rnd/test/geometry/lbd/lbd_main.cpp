#include "../lbd_lib/EDLineDetector.hh"
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <list>


namespace {
namespace local {

void edline_detector_example()
{
	std::list<std::string> image_filepaths;
	image_filepaths.push_back("../data/feature_analysis/chairs.pgm");
	image_filepaths.push_back("../data/feature_analysis/urban_1.jpg");
	image_filepaths.push_back("../data/feature_analysis/urban_2.jpg");
	image_filepaths.push_back("../data/feature_analysis/urban_3.jpg");

	EDLineDetector detector;
	//std::cout << "Min. line length = " << detector.minLineLen_ << std::endl;

	for (const auto &img_fpath : image_filepaths)
	{
		const cv::Mat img(cv::imread(img_fpath, cv::IMREAD_COLOR));
		if (img.empty())
		{
			std::cout << "Failed to load an image file: " << img_fpath << std::endl;
			continue;
		}

		cv::Mat gray;
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

		// Detect lines.
		{
			boost::timer::auto_cpu_timer timer;
			if (!detector.EDline(gray, false))
			{
				std::cerr << "Failed to detect any line by EDLine." << std::endl;
				continue;
			}
		}

		// Show the result.
		std::cout << "\t#detected lines whose length are larger than minLineLen = " << detector.lines_.numOfLines << std::endl;
		std::cout << "\t#x-coords = " << detector.lines_.xCors.size() << std::endl;
		std::cout << "\t#y-coords = " << detector.lines_.yCors.size() << std::endl;
		std::cout << "\t#start indices = " << detector.lines_.sId.size() << std::endl;

		cv::Mat rgb;
		img.copyTo(rgb);
		for (const auto &coords : detector.lineEndpoints_)
		{
			const cv::Point pt1((int)std::round(coords[0]), (int)std::round(coords[1]));
			const cv::Point pt2((int)std::round(coords[2]), (int)std::round(coords[3]));

			cv::line(rgb, pt1, pt2, CV_RGB(255, 0, 0), 1, cv::LINE_AA);
		}

		cv::imshow("EDLines - Detected", rgb);
		cv::waitKey(0);
	}

	cv::destroyAllWindows();
}

void lbd_example()
{
	throw std::runtime_error("Not yet implemented");
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
		//	REF [paper] >> "An efficient and robust line segment matching approach based on LBD descriptor and pairwise geometric consistency", JVCIR 2012.
		//	REF [site] >> https://github.com/mtamburrano/LBD_Descriptor
		//		Basic Image AlgorithmS Library (BIAS) library is required. (?)
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
