#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_itk {

void geodesic_active_contour_example();

}  // namespace my_itk

int itk_main(int argc, char *argv[])
{
	try
	{
		std::cout << "Segmentation --------------------------------------------------------" << std::endl;
		my_itk::geodesic_active_contour_example();
	}
	catch (const itk::ExceptionObject &ex)
	{
		std::cout << "itk::ExceptionObject caught: " << ex.what() << std::endl;
		return 1;
	}

	return 0;
}
