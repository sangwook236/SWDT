//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char *argv[])
{
	int line_feature_main(int argc, char *argv[]);
	int surf_main(int argc, char *argv[]);
	int siftgpu_main(int argc, char *argv[]);
	int sift_main(int argc, char *argv[]);
	int pictorial_structure_main(int argc, char *argv[]);

	try
	{
		// line feature
		//line_feature_main(argc, argv);

		//
		surf_main(argc, argv);
		siftgpu_main(argc, argv);
		sift_main(argc, argv);
		
		// pictorial structure
		pictorial_structure_main(argc, argv);
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
