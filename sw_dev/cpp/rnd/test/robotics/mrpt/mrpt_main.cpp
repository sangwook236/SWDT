//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_mrpt {

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

}  // namespace my_mrpt

int mrpt_main(int argc, char *argv[])
{
	//my_mrpt::feature_extraction_and_matching();
	//my_mrpt::icp();
	//my_mrpt::ransac();

	//my_mrpt::rawlog();
	//my_mrpt::rawlog_grabber();

	my_mrpt::map_handling();
	//my_mrpt::localization_pf();
	//my_mrpt::slam_kf();
	//my_mrpt::slam_icp();

	//my_mrpt::dijkstra();
	//my_mrpt::path_planning();

	return 0;
}
