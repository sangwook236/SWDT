//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <opencv2/text.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <locale>
#include <codecvt>
#include <cassert>


namespace {
namespace local {

void groups_draw(cv::Mat &src, std::vector<cv::Rect> &groups)
{
	for (int i = (int)groups.size() - 1; i >= 0; --i)
	{
		if (src.type() == CV_8UC3)
			cv::rectangle(src, groups.at(i).tl(), groups.at(i).br(), cv::Scalar(0, 255, 255), 3, cv::LINE_AA);
		else
			cv::rectangle(src, groups.at(i).tl(), groups.at(i).br(), cv::Scalar(255), 3, cv::LINE_AA);
	}
}

void er_show(std::vector<cv::Mat> &channels, std::vector<std::vector<cv::text::ERStat>> &regions)
{
	for (int c = 0; c < (int)channels.size(); ++c)
	{
		cv::Mat dst = cv::Mat::zeros(channels[0].rows + 2, channels[0].cols + 2, CV_8UC1);
		for (int r = 0; r < (int)regions[c].size(); ++r)
		{
			cv::text::ERStat er = regions[c][r];
			if (er.parent != NULL) // deprecate the root region
			{
				int newMaskVal = 255;
				int flags = 4 + (newMaskVal << 8) + cv::FLOODFILL_FIXED_RANGE + cv::FLOODFILL_MASK_ONLY;
				cv::floodFill(channels[c], dst, cv::Point(er.pixel%channels[c].cols, er.pixel / channels[c].cols),
					cv::Scalar(255), 0, cv::Scalar(er.level), cv::Scalar(0), flags);
			}
		}
		char buff[20];
		char *buff_ptr = buff;
		sprintf(buff, "channel %d", c);
		cv::imshow(buff_ptr, dst);
	}
}

// REF [file] >>
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/textdetection.cpp
//	${OPENCV_CONTRIB_HOME}/modules/text/samples/detect_er_chars.py
void ER_detector_example()
{
	const std::string img_filepath("D:/lib_repo/cpp/rnd/opencv_contrib_github/modules/text/samples/scenetext_segmented_word04.jpg");

	cv::Mat img = cv::imread(img_filepath, cv::IMREAD_COLOR);
	//cv::Mat img = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);
	if (img.empty())
	{
		std::cerr << "File not found: " << img_filepath << std::endl;
		return;
	}

	// Extract channels to be processed individually
	std::vector<cv::Mat> channels;
	cv::text::computeNMChannels(img, channels);

	int cn = (int)channels.size();
	// Append negative channels to detect ER- (bright regions over dark background)
	for (int c = 0; c < cn - 1; c++)
		channels.push_back(255 - channels[c]);

	// Create ERFilter objects with the 1st and 2nd stage default classifiers
	cv::Ptr<cv::text::ERFilter> er_filter1 = cv::text::createERFilterNM1(cv::text::loadClassifierNM1("D:/lib_repo/cpp/rnd/opencv_contrib_github/modules/text/samples/trained_classifierNM1.xml"), 16, 0.00015f, 0.13f, 0.2f, true, 0.1f);
	cv::Ptr<cv::text::ERFilter> er_filter2 = cv::text::createERFilterNM2(cv::text::loadClassifierNM2("D:/lib_repo/cpp/rnd/opencv_contrib_github/modules/text/samples/trained_classifierNM2.xml"), 0.5);

	std::vector<std::vector<cv::text::ERStat> > regions(channels.size());
	// Apply the default cascade classifier to each independent channel (could be done in parallel).
	std::cout << "Extracting Class Specific Extremal Regions from " << (int)channels.size() << " channels ..." << std::endl;
	std::cout << "    (...) this may take a while (...)" << std::endl << std::endl;
	for (int c = 0; c < (int)channels.size(); ++c)
	{
		er_filter1->run(channels[c], regions[c]);
		er_filter2->run(channels[c], regions[c]);
	}

	// Detect character groups.
	std::cout << "Grouping extracted ERs ... ";
	std::vector<std::vector<cv::Vec2i> > region_groups;
	std::vector<cv::Rect> groups_boxes;
	cv::text::erGrouping(img, channels, regions, region_groups, groups_boxes, cv::text::ERGROUPING_ORIENTATION_HORIZ);
	//cv::text::erGrouping(img, channels, regions, region_groups, groups_boxes, cv::text::ERGROUPING_ORIENTATION_ANY, "./trained_classifier_erGrouping.xml", 0.5);

	// Draw groups.
	groups_draw(img, groups_boxes);
	cv::imshow("grouping", img);

	std::cout << "Done!" << std::endl << std::endl;
	std::cout << "Press 'space' to show the extracted Extremal Regions, any other key to exit." << std::endl << std::endl;
	if ((cv::waitKey() & 0xff) == ' ')
		er_show(channels, regions);

	// Clean-up memory.
	er_filter1.release();
	er_filter2.release();
	regions.clear();
	if (!groups_boxes.empty())
	{
		groups_boxes.clear();
	}

	cv::destroyAllWindows();
}

void decode(const cv::Mat &scores, const cv::Mat &geometry, float scoreThresh, std::vector<cv::RotatedRect> &detections, std::vector<float> &confidences)
{
    detections.clear();
    CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
    CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
    CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);

    const int height = scores.size[2];
    const int width = scores.size[3];
    for (int y = 0; y < height; ++y)
    {
        const float *scoresData = scores.ptr<float>(0, 0, y);
        const float *x0_data = geometry.ptr<float>(0, 0, y);
        const float *x1_data = geometry.ptr<float>(0, 1, y);
        const float *x2_data = geometry.ptr<float>(0, 2, y);
        const float *x3_data = geometry.ptr<float>(0, 3, y);
        const float *anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x)
        {
            float score = scoresData[x];
            if (score < scoreThresh)
                continue;

            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

			cv::Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
			cv::Point2f p1 = cv::Point2f(-sinA * h, -cosA * h) + offset;
			cv::Point2f p3 = cv::Point2f(-cosA * w, sinA * w) + offset;
			cv::RotatedRect r(0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }
}

// REF [file] >> ${OPENCV_HOME}/samples/dnn/text_detection.cpp
void EAST_detector_example()
{
	const float confThreshold = 0.5;  // Confidence threshold.
	const float nmsThreshold = 0.4;  // Non-maximum suppression threshold.
	const int inpWidth = 320;  // Preprocess input image by resizing to a specific width. It should be multiple by 32.
	const int inpHeight = 320;  // Preprocess input image by resizing to a specific height. It should be multiple by 32.
	// REF [file] >> ${OPENCV_HOME}/samples/dnn/frozen_east_text_detection.pb
	const std::string model_filepath("D:/lib_repo/cpp/rnd/opencv_contrib_github/samples/dnn/frozen_east_text_detection.pb");  // Path to a binary.pb file contains trained network.

	if (model_filepath.empty())
	{
		std::cerr << "Failed to load a model file: " << model_filepath << std::endl;
		return;
	}

	const std::string img_filepath("D:/lib_repo/cpp/rnd/opencv_contrib_github/modules/text/samples/scenetext_segmented_word04.jpg");

	cv::Mat img = cv::imread(img_filepath, cv::IMREAD_COLOR);
	if (img.empty())
	{
		std::cerr << "File not found: " << img_filepath << std::endl;
		return;
	}

	// Load network.
	cv::dnn::Net net = cv::dnn::readNet(model_filepath);

	const std::vector<std::string> outNames = {
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3",
	};

	cv::Mat blob;
	cv::dnn::blobFromImage(img, blob, 1.0, cv::Size(inpWidth, inpHeight), cv::Scalar(123.68, 116.78, 103.94), true, false);
	net.setInput(blob);
	std::vector<cv::Mat> outs;
	net.forward(outs, outNames);

	const cv::Mat &scores = outs[0];
	const cv::Mat &geometry = outs[1];

	// Decode predicted bounding boxes.
	std::vector<cv::RotatedRect> boxes;
	std::vector<float> confidences;
	decode(scores, geometry, confThreshold, boxes, confidences);

	// Apply non-maximum suppression procedure.
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	// Render detections.
	const cv::Point2f ratio((float)img.cols / inpWidth, (float)img.rows / inpHeight);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		cv::RotatedRect& box = boxes[indices[i]];

		cv::Point2f vertices[4];
		box.points(vertices);
		for (int j = 0; j < 4; ++j)
		{
			vertices[j].x *= ratio.x;
			vertices[j].y *= ratio.y;
		}
		for (int j = 0; j < 4; ++j)
			line(img, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 1);
	}

	// Put efficiency information.
	std::vector<double> layersTimes;
	double freq = cv::getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	std::string label = cv::format("Inference time: %.2f ms", t);
	putText(img, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

	cv::imshow("EAST", img);
	cv::waitKey(0);

	cv::destroyAllWindows();
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
	//local::ER_detector_example();  // AABB.
	//local::EAST_detector_example();  // OBB.
	//local::TextBoxes_detector_example();  // Not yet implemented.

	//local::HMM_decoder_example();
	//local::beam_search_decoder_example();
	local::tesseract_example();
}

}  // namespace my_opencv
