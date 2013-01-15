//include "stdafx.h"
#include <pabod/handlerOpenCVStructs.h>
#include <pabod/makeDetection.h>
#include <pabod/pabod.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/core_c.h>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <iomanip>
#include <string>


namespace {
namespace local {

// [ref] ${LibPaBOD_HOME}/doc/quickstart.pdf
void simple_example()
{
	//const std::string model_filename("object_detection_data\\libpabod\\models\\bicycle_v6.mat");
	//const std::string model_filename("object_detection_data\\libpabod\\models\\car_v6.mat");
	//const std::string model_filename("object_detection_data\\libpabod\\models\\horse_v6.mat");
	const std::string model_filename("object_detection_data\\libpabod\\models\\person_v6.mat");

	//const std::string input_filename("object_detection_data\\libpabod\\2007_000572.jpg");  // bicycle
	//const std::string input_filename("object_detection_data\\libpabod\\2007_000584.jpg");  // bicycle
	//const std::string input_filename("object_detection_data\\libpabod\\2007_000793.jpg");  // bicycle, person
	//const std::string input_filename("object_detection_data\\libpabod\\2007_001311.jpg");  // bicycle, person
	const std::string input_filename("object_detection_data\\libpabod\\2007_003022.jpg");  // horse, person
	//const std::string input_filename("object_detection_data\\libpabod\\2007_004830.jpg");  // car
	//const std::string input_filename("object_detection_data\\libpabod\\2008_006317.jpg");  // horse, person
	//const std::string input_filename("object_detection_data\\libpabod\\2008_007537.jpg");  // person
	//const std::string input_filename("object_detection_data\\libpabod\\2009_004664.jpg");  // bicycle, person
	//const std::string input_filename("object_detection_data\\libpabod\\2009_004848.jpg");  // bicycle, person
	//const std::string input_filename("object_detection_data\\libpabod\\ucobikes.jpg");  // bicycle

	//
	const std::string output_filename("object_detection_data\\libpabod\\detection_output.jpg");

	std::cout << "loading an input file ..." << std::endl;
	IplImage *img = cvLoadImage(input_filename.c_str(), CV_LOAD_IMAGE_COLOR);

	std::cout << "loading a model file ..." << std::endl;
	Pabod detector(model_filename.c_str());

	//
	std::cout << "start detecting ..." << std::endl;
	boost::timer::cpu_timer timer;

	const float threshold = POSITIVE_INF;
	CvMat *detection_result = NULL;
	const float usedThreshold = detector.detect(img, threshold, &detection_result);

	const boost::timer::cpu_times const elapsed_times(timer.elapsed());
	std::cout << "elapsed times: " << std::fixed << ((elapsed_times.system + elapsed_times.user) * 1e-9) << " sec" << std::endl;
	std::cout << "end detecting ..." << std::endl;

	//
	detector.drawDetections(img, detection_result);

	std::cout << "saving input files ..." << std::endl;
	cvSaveImage(output_filename.c_str(), img);

	cvReleaseImage(&img);
}

}  // namespace local
}  // unnamed namespace

namespace my_libpabod {

}  // namespace my_libpabod

int libpabod_main(int argc, char *argv[])
{
	local::simple_example();

	return 0;
}
