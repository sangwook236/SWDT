//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char *argv[])
{
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

	try
	{
		//feature_extraction_and_matching();
		//icp();
		//ransac();

		//rawlog();
		//rawlog_grabber();

		map_handling();
		//localization_pf();
		//slam_kf();
		//slam_icp();

		//dijkstra();
		//path_planning();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		std::cout << "unknown exception caught" << std::endl;
		return -1;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return 0;
}
