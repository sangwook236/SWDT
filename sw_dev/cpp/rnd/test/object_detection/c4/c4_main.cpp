//include "stdafx.h"
#include "../c4_lib/Pedestrian.h"
#include <boost/timer/timer.hpp>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>


void LoadCascade(DetectionScanner &ds);
int DetectHuman(const char *filename, const char *outname, DetectionScanner &ds, std::ofstream &out);
extern DetectionScanner scanner;
extern int totaltime;

namespace {
namespace local {

// [ref] RunFiles() in ${C4_HOME}/Pedestrian_ICRA.cpp
void simple_example()
{
	// Run detection on a set of files, assuming
	// 1) file listed in "in_base_dir"/"input_file_list", one file name per line
	// 2) output detection results to "out_base_dir" -- create this directory before running the detector
	const std::string in_base_dir("./object_detection_data/");
	const std::string out_base_dir("./object_detection_data/c4/");
	const std::string input_file_list("./object_detection_data/input_file_list.txt");
	const std::string out_filename("./object_detection_data/c4/result_HIK.txt");

	std::ifstream in(input_file_list.c_str());
	std::ofstream out(out_filename.c_str());
	if (!in.is_open() || !out.is_open())
	{
		std::cerr << "input and/or output files cannot be loaded." << std::endl;
		return;
	}

	// load classifiers: combined.txt.model & combined2.txt.model.
	std::cout << "loading detectors ..." << std::endl;
	{
		boost::timer::auto_cpu_timer timer;
		LoadCascade(scanner);
	}
	std::cout << "detectors loaded." << std::endl;

	std::size_t num_files = 0;
	totaltime = 0;
	std::string filename;
	while (in.good())
	{
		in >> filename;

		std::ostringstream buf1, buf2;
		buf1 << in_base_dir << filename;
		buf2 << out_base_dir << filename << ".out";

		std::cout << "start detecting ..." << std::endl;
		{
			boost::timer::auto_cpu_timer timer;

			//DetectHuman(buf1.str().c_str(), NULL, scanner, out);
			// if you want to save detection results, use the next line
			DetectHuman(buf1.str().c_str(), buf2.str().c_str(), scanner, out);
		}
		std::cout << "end detecting ..." << std::endl;

		++num_files;

		cv::waitKey(0);
	}

	std::cout << "processed " << num_files << " images in " << totaltime << " milliseconds." << std::endl;

	in.close();
	out.close();
}

}  // namespace local
}  // unnamed namespace

namespace my_c4 {

}  // namespace my_c4

int c4_main(int argc, char *argv[])
{
	local::simple_example();

	return 0;
}
