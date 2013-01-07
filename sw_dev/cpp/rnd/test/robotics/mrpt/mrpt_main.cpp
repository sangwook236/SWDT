//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace mrpt {

void feature_extraction_and_matching();
void icp();
void ransac();

void rawlog();
void rawlog_grabber();

void map_handling();
void localization_pf();
void slam_kf();
void slam_icp();

void dijkstra();
void path_planning();

}  // namespace mrpt

int mrpt_main(int argc, char *argv[])
{
	//mrpt::feature_extraction_and_matching();
	//mrpt::icp();
	//mrpt::ransac();

	//mrpt::rawlog();
	//mrpt::rawlog_grabber();

	mrpt::map_handling();
	//mrpt::localization_pf();
	//mrpt::slam_kf();
	//mrpt::slam_icp();

	//mrpt::dijkstra();
	//mrpt::path_planning();

	return 0;
}
