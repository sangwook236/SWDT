//#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opensift {

void extract_feature();
void display_feature();
void match_feature();

}  // namespace my_opensift

int opensift_main(int argc, char *argv[])
{
	try
	{
		my_opensift::extract_feature();
		//my_opensift::display_feature();
		//my_opensift::match_feature();
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
