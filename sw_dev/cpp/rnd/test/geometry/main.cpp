#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


int main(int argc, char *argv[])
{
	int lsd_main(int argc, char* argv[]);
	int lbd_main(int argc, char* argv[]);
	int elsd_main(int argc, char* argv[]);

	int pcl_main(int argc, char *argv[]);
	int open3d_main(int argc, char *argv[]);
	int threedtk_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::cout << "Line feature --------------------------------------------------------" << std::endl;
		//retval = lsd_main(argc, argv);  // Line segment detector (LSD).
		retval = lbd_main(argc, argv);  // EDLine detector & line band descriptor (LBD).

		std::cout << "\nEllipse & line feature ----------------------------------------------" << std::endl;
		//retval = elsd_main(argc, argv);

		std::cout << "\nPoint Cloud Library (PCL) -------------------------------------------" << std::endl;
		//retval = pcl_main(argc, argv);

		std::cout << "\nOpen3D --------------------------------------------------------------" << std::endl;
		//retval = open3d_main(argc, argv);

		std::cout << "\n3DTK - The 3D Toolkit -----------------------------------------------" << std::endl;
		//	- 3D point clouds.
		//retval = threedtk_main(argc, argv);  // Not yet implemented.
	}
	catch (const std::bad_alloc &ex)
	{
		std::cout << "std::bad_alloc caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &ex)
	{
		std::cout << "std::exception caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
