#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void resampling();
void greedy_projection();
void ransac();
void visualization(int argc, char **argv);

}  // namespace my_pcl

int pcl_main(int argc, char *argv[])
{
	// Tutorials -----------------------------------------------------------
	//	REF [site] >> http://pointclouds.org/documentation/tutorials/

	//my_pcl::resampling();
	//my_pcl::greedy_projection();

	my_pcl::ransac();

	//my_pcl::visualization(argc, argv);

	return 0;
}
