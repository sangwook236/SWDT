#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void resampling();
void greedy_projection();
void visualization(int argc, char **argv);

}  // namespace my_pcl

int pcl_main(int argc, char *argv[])
{
	// tutorials -----------------------------------------------------------
	//my_pcl::resampling();  // not correctly working.
	my_pcl::greedy_projection();

	//my_pcl::visualization(argc, argv);

	return 0;
}
