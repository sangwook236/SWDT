#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace sift {

void extract_sift_feature();
void display_sift_feature();
void match_sift_feature();

}  // namespace sift

int sift_main(int argc, char *argv[])
{
	try
	{
		//sift::extract_sift_feature();
		//sift::display_sift_feature();
		sift::match_sift_feature();
	}
	catch (const cv::Exception &e)
	{
		//std::cout << "OpenCV exception occurred !!!: " << e.what() << std::endl;
		//std::cout << "OpenCV exception occurred !!!: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception occurred !!!:" << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;
	}

	return 0;
}
