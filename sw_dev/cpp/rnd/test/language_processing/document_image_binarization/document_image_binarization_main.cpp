//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>


// [ref] in binarizewolfjolion.cpp.
enum NiblackVersion 
{
	NIBLACK=0,
    SAUVOLA,
    WOLFJOLION,
};
void NiblackSauvolaWolfJolion(cv::Mat im, cv::Mat output, NiblackVersion version, int winx, int winy, double k, double dR);

namespace {
namespace local {

// [ref] http://liris.cnrs.fr/christian.wolf/software/binarize/.
void niblack_sauvola_wolfjolion_algoirthm()
{
	const NiblackVersion versionCode = NIBLACK;  // NIBLACK, SAUVOLA, WOLFJOLION.
	//const std::string inputname = "./data/language_processing/document_image_binarization/sample.jpg";
	const std::string inputname = "./data/language_processing/document_image_binarization/sample_gray.jpg";
	//const std::string inputname = "./data/language_processing/document_image_binarization/sample_gray_inverted.jpg";
	const std::string outputname = "./data/language_processing/document_image_binarization/sample_out.jpg";
	
	//std::cout << "BINARIZEWOLF Version " << BINARIZEWOLF_VERSION << std::endl;

    // Load the image in grayscale mode
    cv::Mat input = cv::imread(inputname, CV_LOAD_IMAGE_GRAYSCALE);
    if ((input.rows <= 0) || (input.cols <= 0))
	{
        std::cerr << "input image error: " << inputname << std::endl;
        return;
    }

	const float optK = 0.5f;
    // the window size.
    int winy = (int)(2.0 * input.rows - 1) / 3;
    int winx = ((int)input.cols - 1 < winy) ? (input.cols - 1) : winy;
    // if the window is too big, then we assume that the image is not a single text box, but a document page.
	// set the window size to a fixed constant.
    if (winx > 100)
        winx = winy = 40;
    
    // Threshold.
    cv::Mat output(input.rows, input.cols, CV_8U);
    NiblackSauvolaWolfJolion(input, output, versionCode, winx, winy, optK, 128);

    // Write the tresholded file.
    std::cout << "Writing binarized image to file: " << outputname << std::endl;
    cv::imwrite(outputname, output);
}

}  // namespace local
}  // unnamed namespace

namespace my_document_image_binarization {

}  // namespace my_document_image_binarization

int document_image_binarization_main(int argc, char *argv[])
{
	local::niblack_sauvola_wolfjolion_algoirthm();

	return 0;
}

