//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
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

	int quirc_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::cout << "OpenCV library ------------------------------------------------------" << std::endl;
		retval = opencv_main(argc, argv);

		std::cout << "\nVXL (the Vision-something-Libraries) library ------------------------" << std::endl;
		//	- Pictorial structures matching.
		//	- Shape model library.
		//retval = vxl_main(argc, argv);

		std::cout << "\nIntegrating Vision Toolkit (IVT) library ----------------------------" << std::endl;
		//	- Hough transform.
		//	- KLT tracker.
		//	- Particle filtering.
		//retval = ivt_main(argc, argv);

		std::cout << "\nVision Lab Features Library (VLFeat) --------------------------------" << std::endl;
		//	- Feature analysis.
		//		SIFT, DSIFT, MSER, HOG, LBP.
		//		Covariant detectors.
		//		Local Intensity Order Pattern (LIOP).
		//		Bag of Visual Words (BoVW).
		//		Vector of Locally Aggregated Descriptors (VLAD) encoding, Fisher Vector (FV) encoding.
		//	- Clustering.
		//		k-means, GMM, IKM, HIKM, AIB.
		//	- Segmentation.
		//		Quick shift, SLIC.
		//	- Learning algorithm.
		//		SVM.
		//retval = vlfeat_main(argc, argv);

		std::cout << "\nC-based/Cached/Core Computer Vision (CCV) library -------------------" << std::endl;
		//	- Feature analysis.
		//		Scale invariant feature transform (SIFT).
		//		DAISY.
		//		Histogram of oriented gradients (HOG).
		//		Brightness binary feature (BBF).
		//	- Object detection & tracking.
		//		Deformable parts model (DPM) - discriminatively trained part-based models.
		//		Stroke width transform (SWT).
		//		Tracking-learning-detection (TLD).
		//	- Sparse coding & compressive sensing.
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		// TODO [check] >> run-time error in Windows. not correctly working.
		//retval = ccv_main(argc, argv);
#else
        std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif

		std::cout << "\nVision with Generic Algorithms (VIGRA) library ----------------------" << std::endl;
		//	- Segmentation.
		//		Seeded region growing (SRG).
		//		Watershed region growing.
		//		Simple linear iterative clustering (SLIC).
		//retval = vigra_main(argc, argv);

		std::cout << "\nDarwin library ------------------------------------------------------" << std::endl;
		//	- Machine learning.
		//	- Probabilistic graphical model.
		//retval = darwin_main(argc, argv);  // Not yet implemented.

		std::cout << "\nquirc library -------------------------------------------------------" << std::endl;
		//	- QR code.
		//retval = quirc_main(argc, argv);
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
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
