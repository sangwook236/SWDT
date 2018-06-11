//include "stdafx.h"
#include <pabod/handlerOpenCVStructs.h>
#include <pabod/makeDetection.h>
#include <pabod/pabod.h>
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <iomanip>
#include <string>


namespace {
namespace local {

// REF [doc] >> ${LibPaBOD_HOME}/doc/quickstart.pdf
void simple_example()
{
	//const std::string model_filename("./data/object_detection/libpabod/models/bicycle_v6.mat");
	//const std::string model_filename("./data/object_detection/libpabod/models/car_v6.mat");
	//const std::string model_filename("./data/object_detection/libpabod/models/horse_v6.mat");
	const std::string model_filename("./data/object_detection/libpabod/models/person_v6.mat");

	//const std::string input_filename("./data/object_detection/2007_000572.jpg");  // Bicycle.
	//const std::string input_filename("./data/object_detection/2007_000584.jpg");  // Bicycle.
	//const std::string input_filename("./data/object_detection/2007_000793.jpg");  // Bicycle, person.
	//const std::string input_filename("./data/object_detection/2007_001311.jpg");  // Bicycle, person.
	const std::string input_filename("./data/object_detection/2007_003022.jpg");  // Horse, person.
	//const std::string input_filename("./data/object_detection/2007_004830.jpg");  // Car.
	//const std::string input_filename("./data/object_detection/2008_006317.jpg");  // Horse, person.
	//const std::string input_filename("./data/object_detection/2008_007537.jpg");  // Person.
	//const std::string input_filename("./data/object_detection/2009_004664.jpg");  // Bicycle, person.
	//const std::string input_filename("./data/object_detection/2009_004848.jpg");  // Bicycle, person.
	//const std::string input_filename("./data/object_detection/ucobikes.jpg");  // Bicycle.

	//
	const std::string output_filename("./data/object_detection/libpabod/detection_output.jpg");

	std::cout << "Loading an input file ..." << std::endl;
	IplImage *img = cvLoadImage(input_filename.c_str(), cv::IMREAD_COLOR);

	std::cout << "Loading a model file ..." << std::endl;
	Pabod detector(model_filename.c_str());

	//
	const float threshold = POSITIVE_INF;
	CvMat *detection_result = NULL;

	std::cout << "Start detecting ..." << std::endl;
	{
        boost::timer::cpu_timer timer;

        const float usedThreshold = detector.detect(img, threshold, &detection_result);

        const boost::timer::cpu_times elapsed_times(timer.elapsed());
        std::cout << "Elapsed times: " << std::fixed << ((elapsed_times.system + elapsed_times.user) * 1e-9) << " sec" << std::endl;
	}
	std::cout << "End detecting ..." << std::endl;

	//
	detector.drawDetections(img, detection_result);

	std::cout << "Saving input files ..." << std::endl;
	cvSaveImage(output_filename.c_str(), img);

	cvReleaseImage(&img);
}

}  // namespace local
}  // unnamed namespace

namespace my_libpabod {

}  // namespace my_libpabod

int libpabod_main(int argc, char *argv[])
{
	// REF [paper] >> "Object Detection with Discriminatively Trained Part-Based Models", P. Felzenszwalb, R. Girshick, D. McAllester, & D. Ramanan, TPAMI, 2010.

	local::simple_example();

	return 0;
}
