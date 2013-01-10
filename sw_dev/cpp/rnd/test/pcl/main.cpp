#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


namespace my_pcl {

void resampling();
void greedy_projection();
void pcl_visualization(int argc, char **argv);

}  // namespace my_pcl

int main(int argc, char *argv[])
{
	try
	{
		// tutorials
		//my_pcl::resampling();
		my_pcl::greedy_projection();

		//my_pcl::pcl_visualization(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

    return 0;
}
