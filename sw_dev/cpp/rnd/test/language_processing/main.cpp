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
	int tesseract_main(int argc, char *argv[]);
	int leptonica_main(int argc, char *argv[]);

	int document_image_binarization_main(int argc, char *argv[]);

	int kaldi_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::cout << "Tesseract OCR engine ------------------------------------------------" << std::endl;
		//retval = tesseract_main(argc, argv);  // not yet implemented
		std::cout << "\nLeptonica Image Processing Library ----------------------------------" << std::endl;
		//	-. OCR.
		//retval = leptonica_main(argc, argv);  // not yet implemented

		std::cout << "\nDocument Image Binarization algorithm -------------------------------" << std::endl;
		retval = document_image_binarization_main(argc, argv);

		std::cout << "\nKaldi Project -------------------------------------------------------" << std::endl;
		//	-. a toolkit for speech recognition.
		//retval = kaldi_main(argc, argv);  // not yet implemented
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
