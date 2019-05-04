//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/text.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <locale>
#include <codecvt>
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
	// Must have the same order as the classifier output classes.
	const std::string vocabulary("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");

	//cv::Ptr<cv::text::OCRHMMDecoder::ClassifierCallback> ocr = cv::text::loadOCRHMMClassifierNM("./OCRHMM_knn_model_data.xml.gz");
	cv::Ptr<cv::text::OCRHMMDecoder::ClassifierCallback> ocr = cv::text::loadOCRHMMClassifierCNN("./OCRBeamSearch_CNN_model_data.xml.gz");

	const std::vector<std::string> image_filepaths = {
	};

	std::vector<int> out_classes;
	std::vector<double> out_confidences;
	for (const auto& image_filepath : image_filepaths)
	{
		cv::Mat &image = cv::imread(image_filepath, cv::IMREAD_COLOR);
		if (image.empty())
		{
			std::cerr << "Failed to load an image: " << image_filepath << std::endl;
			continue;
		}

		const double t_r = (double)cv::getTickCount();

		ocr->eval(image, out_classes, out_confidences);

		std::cout << "OCR output = \"" << vocabulary[out_classes[0]] << "\" with confidence "
			<< out_confidences[0] << ". Evaluated in "
			<< ((double)cv::getTickCount() - t_r) * 1000 / cv::getTickFrequency() << " ms." << std::endl << std::endl;
	}
}

// REF [file] >> ${OPENCV_CONTRIB_HOME}/modules/text/samples/cropped_word_recognition.cpp
void beam_search_decoder_example()
{
	// Must have the same order as the classifier output classes.
	std::string vocabulary("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");

	cv::Mat transition_p;
#if true
	// A list of words expected to be found on the input image.
	std::vector<std::string> lexicon;
	lexicon.push_back(std::string("abb"));
	lexicon.push_back(std::string("riser"));
	lexicon.push_back(std::string("CHINA"));
	lexicon.push_back(std::string("HERE"));
	lexicon.push_back(std::string("President"));
	lexicon.push_back(std::string("smash"));
	lexicon.push_back(std::string("KUALA"));
	lexicon.push_back(std::string("Produkt"));
	lexicon.push_back(std::string("NINTENDO"));

	// Create tailored language model a small given lexicon,
	cv::text::createOCRHMMTransitionsTable(vocabulary, lexicon, transition_p);
#else
	// Load the default generic language model (created from ispell 42869 English words list).
	cv::FileStorage fs("./OCRHMM_transitions_table.xml", cv::FileStorage::READ);
	fs["transition_probabilities"] >> transition_p;
	fs.release();
#endif

	const cv::Mat emission_p(cv::Mat::eye(62, 62, CV_64FC1));

	// Notice we set here a beam size of 50.
	// This is much faster than using the default value (500).
	// 50 works well with our tiny lexicon example, but may not with larger dictionaries.
	cv::Ptr<cv::text::OCRBeamSearchDecoder> ocr = cv::text::OCRBeamSearchDecoder::create(
		cv::text::loadOCRBeamSearchClassifierCNN("./OCRBeamSearch_CNN_model_data.xml.gz"),
		vocabulary, transition_p, emission_p, cv::text::OCR_DECODER_VITERBI, 50
	);

	const std::vector<std::string> image_filepaths = {
	};

	std::string output;
	std::vector<cv::Rect> boxes;
	std::vector<std::string> words;
	std::vector<float> confidences;
	for (const auto& image_filepath : image_filepaths)
	{
		cv::Mat &image = cv::imread(image_filepath, cv::IMREAD_COLOR);
		if (image.empty())
		{
			std::cerr << "Failed to load an image: " << image_filepath << std::endl;
			continue;
		}

		const double t_r = (double)cv::getTickCount();

		ocr->run(image, output, &boxes, &words, &confidences, cv::text::OCR_LEVEL_WORD);

		std::cout << "OCR output = \"" << output << "\". Decoded in "
			<< ((double)cv::getTickCount() - t_r) * 1000 / cv::getTickFrequency() << " ms." << std::endl << std::endl;
	}
}

// REF [file] >>
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/segmented_word_recognition.cpp
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/end_to_end_recognition.cpp
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/webcam_demo.cpp
void tesseract_example()
{
	// Character recognition vocabulary.
	// Must have the same order as the classifier output classes.
	std::string vocabulary("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
	// Emission probabilities for the HMM language model (identity matrix by default).
	const cv::Mat emissionProbabilities(cv::Mat::eye((int)vocabulary.size(), (int)vocabulary.size(), CV_64FC1));
	// Bigram transition probabilities for the HMM language model.
	cv::Mat transitionProbabilities;
#if false
	// A list of words expected to be found on the input image.
	std::vector<std::string> lexicon;
	lexicon.push_back(std::string("abb"));
	lexicon.push_back(std::string("riser"));
	lexicon.push_back(std::string("CHINA"));
	lexicon.push_back(std::string("HERE"));
	lexicon.push_back(std::string("President"));
	lexicon.push_back(std::string("smash"));
	lexicon.push_back(std::string("KUALA"));
	lexicon.push_back(std::string("Produkt"));
	lexicon.push_back(std::string("NINTENDO"));

	// Create tailored language model a small given lexicon,
	cv::text::createOCRHMMTransitionsTable(vocabulary, lexicon, transitionProbabilities);
#else
	// Load the default generic language model (created from ispell 42869 English words list).
	cv::FileStorage fs("./OCRHMM_transitions_table.xml", cv::FileStorage::READ);
	fs["transition_probabilities"] >> transitionProbabilities;
	fs.release();
#endif

	//cv::Ptr<cv::text::OCRTesseract> ocrTes = cv::text::OCRTesseract::create();
	const std::string tessdata_dir_path("tessdata_best/");
	const std::string tess_lang("eng+kor");
	const std::string char_whitelist("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");  // Specifies the list of characters used for recognition.
	cv::Ptr<cv::text::OCRTesseract> ocrTes = cv::text::OCRTesseract::create(
		tessdata_dir_path.c_str(), tess_lang.c_str(),
		char_whitelist.c_str(),
		cv::text::OEM_DEFAULT, cv::text::PSM_AUTO
	);
	cv::Ptr<cv::text::OCRHMMDecoder> ocrNM = cv::text::OCRHMMDecoder::create(
		cv::text::loadOCRHMMClassifierNM("./OCRHMM_knn_model_data.xml.gz"),
		vocabulary, transitionProbabilities, emissionProbabilities, cv::text::OCR_DECODER_VITERBI
	);
	cv::Ptr<cv::text::OCRHMMDecoder> ocrCNN = cv::text::OCRHMMDecoder::create(
		cv::text::loadOCRHMMClassifierCNN("./OCRBeamSearch_CNN_model_data.xml.gz"),
		vocabulary, transitionProbabilities, emissionProbabilities, cv::text::OCR_DECODER_VITERBI
	);
	cv::Ptr<cv::text::OCRBeamSearchDecoder> ocrBSCNN = cv::text::OCRBeamSearchDecoder::create(
		cv::text::loadOCRBeamSearchClassifierCNN("./OCRBeamSearch_CNN_model_data.xml.gz"),
		vocabulary, transitionProbabilities, emissionProbabilities, cv::text::OCR_DECODER_VITERBI, 500
	);

	const std::vector<std::pair<std::string, std::string>> image_mask_pairs = {
		std::make_pair(std::string(), std::string()),
	};

	std::string output;
	std::vector<cv::Rect> boxes;
	std::vector<std::string> words;
	std::vector<float> confidences;
	for (const auto& im_pair : image_mask_pairs)
	{
		const std::string image_filepath(im_pair.first);
		const std::string mask_filepath(im_pair.second);

		cv::Mat &image = cv::imread(image_filepath, cv::IMREAD_COLOR);
		if (image.empty())
		{
			std::cerr << "Failed to load an image: " << image_filepath << std::endl;
			continue;
		}
		// Binary segmentation mask where each contour is a character.
		cv::Mat mask;
		if (!mask_filepath.empty())
		{
			mask = cv::imread(mask_filepath, cv::IMREAD_GRAYSCALE);
			if (mask.empty())
				std::cerr << "Failed to load a mask: " << mask_filepath << std::endl;
		}

		// Resize.
		const double height_ratio = 70.0 / image.rows;
		cv::resize(image, image, cv::Size(0, 0), height_ratio, height_ratio, cv::INTER_CUBIC);
		if (!mask.empty())
			cv::resize(mask, mask, cv::Size(0, 0), height_ratio, height_ratio, cv::INTER_CUBIC);

		double t_r = (double)cv::getTickCount();
		if (mask.empty())
			ocrTes->run(image, output, &boxes, &words, &confidences, cv::text::OCR_LEVEL_WORD);
			//ocrTes->run(mask, output, &boxes, &words, &confidences, cv::text::OCR_LEVEL_WORD);
		else
			ocrTes->run(image, mask, output, &boxes, &words, &confidences, cv::text::OCR_LEVEL_WORD);
		output.erase(std::remove(output.begin(), output.end(), '\n'), output.end());
#if true
		std::cout << " OCR_Tesseract  output \"" << output << "\". Done in "
			<< ((double)cv::getTickCount() - t_r) * 1000 / cv::getTickFrequency() << " ms." << std::endl;
#else
		// TODO [check] >> This is not tested.

		std::wcout.imbue(std::locale("kor"));
		std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

		std::wcout << L" OCR_Tesseract  output \"" << conv.from_bytes(output) << L"\". Done in "
			<< ((double)cv::getTickCount() - t_r) * 1000 / cv::getTickFrequency() << L" ms." << std::endl;
#endif

		t_r = (double)cv::getTickCount();
		if (mask.empty())
			ocrNM->run(image, output, &boxes, &words, &confidences, cv::text::OCR_LEVEL_WORD);
			//ocrNM->run(mask, output, &boxes, &words, &confidences, cv::text::OCR_LEVEL_WORD);
		else
			ocrNM->run(image, mask, output, &boxes, &words, &confidences, cv::text::OCR_LEVEL_WORD);
		std::cout << " OCR_NM         output \"" << output << "\". Done in "
			<< ((double)cv::getTickCount() - t_r) * 1000 / cv::getTickFrequency() << " ms." << std::endl;

		t_r = (double)cv::getTickCount();
		if (mask.empty())
			ocrCNN->run(image, output, &boxes, &words, &confidences, cv::text::OCR_LEVEL_WORD);
		else
			ocrCNN->run(image, mask, output, &boxes, &words, &confidences, cv::text::OCR_LEVEL_WORD);
		std::cout << " OCR_CNN        output \"" << output << "\". Done in "
			<< ((double)cv::getTickCount() - t_r) * 1000 / cv::getTickFrequency() << " ms." << std::endl;

		t_r = (double)cv::getTickCount();
		if (mask.empty())
			ocrBSCNN->run(image, output, &boxes, &words, &confidences, cv::text::OCR_LEVEL_WORD);
		else
			ocrBSCNN->run(image, mask, output, &boxes, &words, &confidences, cv::text::OCR_LEVEL_WORD);
		std::cout << " OCR_BS_CNN     output \"" << output << "\". Decoded in "
			<< ((double)cv::getTickCount() - t_r) * 1000 / cv::getTickFrequency() << " ms." << std::endl << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

// REF [site] >>
//	https://docs.opencv.org/4.1.0/da/d56/group__text__detect.html
//	https://docs.opencv.org/4.1.0/d8/df2/group__text__recognize.html

void text()
{
	//local::EAST_detector_example();  // Not yet implemented.
	//local::ER_detector_example();  // Not yet implemented.
	//local::TextBoxes_detector_example();  // Not yet implemented.

	//local::HMM_decoder_example();
	//local::beam_search_decoder_example();
	local::tesseract_example();
}

}  // namespace my_opencv
