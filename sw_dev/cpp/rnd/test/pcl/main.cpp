#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char *argv[])
{
	void resampling();
	void greedy_projection();
	void pcl_visualizion(int argc, char **argv);

	try
	{
		// tutorials
		//resampling();
		greedy_projection();

		//pcl_visualizion(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred !!!: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred !!!" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

    return 0;
}
