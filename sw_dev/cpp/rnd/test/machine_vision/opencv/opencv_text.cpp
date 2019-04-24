//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/text.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>


namespace {
namespace local {

// REF [file] >> ${OPENCV_HOME}/samples/dnn/text_detection.cpp
void EAST_detector_example()
{
	throw std::runtime_error("Not yet implemented");
}

// REF [file] >>
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/textdetection.cpp
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/detect_er_chars.py
void ER_detector_example()
{
	throw std::runtime_error("Not yet implemented");
}

// REF [file] >>
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/deeptextdetection.py
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/textbox_demo.cpp
void TextBoxes_detector_example()
{
	throw std::runtime_error("Not yet implemented");
}

// REF [file] >>
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/character_recognition.cpp
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/segmented_word_recognition.cpp
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/webcam_demo.cpp
void HMM_decoder_example()
{
	throw std::runtime_error("Not yet implemented");
}

// REF [file] >> ${OPENCV_CONTRIB_HOME}/modules/text/samples/cropped_word_recognition.cpp
void beam_search_decoder_example()
{
	throw std::runtime_error("Not yet implemented");
}

// REF [file] >>
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/end_to_end_recognition.cpp
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/segmented_word_recognition.cpp
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/webcam_demo.cpp
void tesseract_example()
{
	throw std::runtime_error("Not yet implemented");
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

// REF [site] >>
//	https://docs.opencv.org/4.1.0/da/d56/group__text__detect.html
//	https://docs.opencv.org/4.1.0/d8/df2/group__text__recognize.html

void text()
{
	local::EAST_detector_example();  // Not yet implemented.
	local::ER_detector_example();  // Not yet implemented.
	local::TextBoxes_detector_example();  // Not yet implemented.

	local::HMM_decoder_example();  // Not yet implemented.
	local::beam_search_decoder_example();  // Not yet implemented.
	local::tesseract_example();  // Not yet implemented.
}

}  // namespace my_opencv
