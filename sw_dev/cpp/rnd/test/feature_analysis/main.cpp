//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char *argv[])
{
	int lsd_main(int argc, char *argv[]);
	int sift_main(int argc, char *argv[]);
	int siftgpu_main(int argc, char *argv[]);
	int surf_main(int argc, char *argv[]);
	int pictorial_structure_revisited_main(int argc, char *argv[]);

	try
	{
		// line feature
		//lsd_main(argc, argv);

		// local descriptor
		//sift_main(argc, argv);
		//siftgpu_main(argc, argv);  // run-time error
		//surf_main(argc, argv);  // run-time error
		
		// pictorial structure
		pictorial_structure_revisited_main(argc, argv);
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
