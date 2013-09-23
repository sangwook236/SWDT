//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


int main(int argc, char *argv[])
{
	int opencv_main(int argc, char *argv[]);
	int vxl_main(int argc, char *argv[]);
	int ivt_main(int argc, char *argv[]);
	int vlfeat_main(int argc, char *argv[]);
	int ccv_main(int argc, char *argv[]);
	int vigra_main(int argc, char *argv[]);
	int darwin_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::cout << "OpenCV library ------------------------------------------------------" << std::endl;
		//retval = opencv_main(argc, argv);

		std::cout << "\nVXL (the Vision-something-Libraries) library ------------------------" << std::endl;
		//	-. pictorial structures matching.
		//	-. shape model library.
		//retval = vxl_main(argc, argv);

		std::cout << "\nIntegrating Vision Toolkit (IVT) library ----------------------------" << std::endl;
		//	-. Hough transform.
		//	-. KLT tracker.
		//	-. particle filtering.
		//retval = ivt_main(argc, argv);

		std::cout << "\nVision Lab Features Library (VLFeat) --------------------------------" << std::endl;
		//	-. feature analysis.
		//		Covariant detectors, HOG, SIFT, DSIFT, MSER.
		//	-. clustering.
		//		k-means, IKM, HIKM, AIB.
		//	-. segmentation.
		//		Quick shift, SLIC.
		retval = vlfeat_main(argc, argv);

		std::cout << "\nC-based/Cached/Core Computer Vision (CCV) library -------------------" << std::endl;
		//	-. feature analysis.
		//		scale invariant feature transform (SIFT).
		//		DAISY.
		//		histogram of oriented gradients (HOG).
		//		brightness binary feature (BBF).
		//	-. object detection & tracking.
		//		deformable parts model (DPM) - discriminatively trained part-based models.
		//		stroke width transform (SWT).
		//		tracking-learning-detection (TLD).
		//	-. sparse coding & compressive sensing.
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		// TODO [check] >> run-time error in Windows. not correctly working.
		//retval = ccv_main(argc, argv);
#endif

		std::cout << "\nVigra library -------------------------------------------------------" << std::endl;
		//	-. segmentation.
		//		seeded region growing (SRG).
		//		watershed region growing.
		//		simple linear iterative clustering (SLIC).
		//retval = vigra_main(argc, argv);

		std::cout << "\nDarwin library ------------------------------------------------------" << std::endl;
		//	-. machine learning.
		//	-. probabilistic graphical model.
		//retval = darwin_main(argc, argv);  // not yet implemented.
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
