#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


namespace my_pcl {

void resampling();
void greedy_projection();
void pcl_visualization(int argc, char **argv);

}  // namespace my_pcl

int main(int argc, char *argv[])
{
	int retval = EXIT_SUCCESS;
	try
	{
		// tutorials
		//my_pcl::resampling();
		my_pcl::greedy_projection();

		//my_pcl::pcl_visualization(argc, argv);
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
