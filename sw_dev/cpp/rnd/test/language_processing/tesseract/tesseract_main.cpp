//#include "stdafx.h"
#include <tesseract/baseapi.h>
//#include <tesseract/strngs.h>
#include <tesseract/renderer.h>
#if defined(__linux) || defined(__linux__) || defined(linux) || defined(__unix) || defined(__unix__) || defined(unix)
#include <tesseract/ocrclass.h>
#endif
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <thread>
#include <list>
#include <array>
#include <chrono>
#include <string>
#include <memory>
#include <locale>
#include <codecvt>


namespace my_leptonica {
	
Pix * mat8ToPix(const cv::Mat &mat8);
cv::Mat pix8ToMat(const Pix *pix8);
cv::Mat pix32ToMat(const Pix *pix32);

}  // namespace my_leptonica

namespace {
namespace local {

void simple_text_recognition_example()
{
	const std::string tessdata_dir_path("../data/language_processing/tessdata/");
	const std::string tesslang("eng+kor");

	tesseract::TessBaseAPI api;
	//if (api.Init(nullptr, tesslang.c_str(), tesseract::OEM_DEFAULT))
	if (api.Init(tessdata_dir_path.c_str(), tesslang.c_str(), tesseract::OEM_DEFAULT))
	{
		std::cerr << "Failed to initialize tesseract." << std::endl;
		return;
	}

	api.SetPageSegMode(tesseract::PSM_AUTO);
	api.SetOutputName("outputbase");

	//const std::string image_filepath("../data/language_processing/phototest.tif");
	//const std::string image_filepath("../data/language_processing/eurotext.tif");
	//const std::string image_filepath("../data/language_processing/receipt.png");
	//const std::string image_filepath("../data/language_processing/road-sign-1.jpg");
	//const std::string image_filepath("../data/language_processing/road-sign-2-768x347.jpg");
	//const std::string image_filepath("../data/language_processing/road-sign-3-300x112.jpg");
	//const std::string image_filepath("../data/language_processing/korean_article_1.png");
	//const std::string image_filepath("../data/language_processing/korean_article_2.png");
	const std::string image_filepath("../data/language_processing/korean_newspaper_1.png");
	//const std::string image_filepath("../data/language_processing/korean_newspaper_2.png");
	//const std::string image_filepath("../data/language_processing/korean_newspaper_3.png");
	//const std::string image_filepath("../data/language_processing/korean_newspaper_4.png");

#if false
	STRING text_out;
	api.ProcessPages(image_filepath.c_str(), nullptr, 0, &text_out);

	std::cout << text_out.string();
#elif true
	const std::string output_base("./test_ocr_results");
	tesseract::TessTextRenderer renderer(output_base.c_str());  // Outputs to "./test_ocr_results.txt".
	//tesseract::TessHOcrRenderer renderer(output_base.c_str());  // Outputs to "./test_ocr_results.hocr".
	//tesseract::TessTsvRenderer renderer(output_base.c_str());  // Outputs to "./test_ocr_results.tsv".
	//tesseract::TessPDFRenderer renderer(output_base.c_str(), tessdata_dir_path.c_str(), true);  // Outputs to "./test_ocr_results.pdf". Needs "tessdata/pdf.ttf".
	//tesseract::TessUnlvRenderer renderer(output_base.c_str());  // Outputs to "./test_ocr_results.unlv".
	//tesseract::TessBoxTextRenderer renderer(output_base.c_str());  // Outputs to "./test_ocr_results.box".
	//tesseract::TessOsdRenderer renderer(output_base.c_str());  // Outputs to "./test_ocr_results.osd".

	std::cout << "Start processing..." << std::endl;
	const auto start_time = std::chrono::high_resolution_clock::now();
	const bool succeed = api.ProcessPages(image_filepath.c_str(), nullptr, 0, &renderer);
	std::cout << "End processing: " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;

	if (!succeed)
		std::cerr << "Error during processing." << std::endl;
#endif
}

// REF [site] >> https://github.com/tesseract-ocr/tesseract/wiki/APIExample
void text_recognition_example()
{
	const std::string tessdata_dir_path("../data/language_processing/tessdata/");
	const std::string tesslang("kor+eng");
	char *configs[] = {
		"my_tesseract.conf"  // tessdata_dir_path + "/configs" + "/my_tesseract.conf".
	};
	const int config_size = 1;
	GenericVector<STRING> vars_vec;
	vars_vec.push_back(STRING("user_defined_dpi"));
	GenericVector<STRING> vars_values;
	vars_values.push_back(STRING("100"));
	const bool set_only_non_debug_params = false;

	tesseract::TessBaseAPI api;
	//if (api.Init(nullptr, nullptr, tesseract::OEM_DEFAULT))
	if (api.Init(tessdata_dir_path.c_str(), tesslang.c_str(), tesseract::OEM_DEFAULT))
	//if (api.Init(tessdata_dir_path.c_str(), tesslang.c_str(), tesseract::OEM_DEFAULT, configs, config_size, nullptr, nullptr, set_only_non_debug_params))
	//if (api.Init(tessdata_dir_path.c_str(), tesslang.c_str(), tesseract::OEM_DEFAULT, configs, config_size, &vars_vec, &vars_values, set_only_non_debug_params))
	{
		std::cerr << "Failed to initialize tesseract." << std::endl;
		return;
	}

	api.SetPageSegMode(tesseract::PSM_AUTO);
	api.SetOutputName("outputbase");

	//api.SetVariable("debug_file", "tesseract.log");  // File to send tprintf() output to. tprintf() in ${TESSERACT_HOME}/src/ccutil/tprintf.h.
	api.SetVariable("user_defined_dpi", "100");
	//api.SetVariable("user_words_file", "eng");
	//api.SetVariable("user_words_suffix", "user-words");
	//api.SetVariable("user_patterns_file", "eng");
	//api.SetVariable("user_patterns_suffix", "user-patterns");
	//api.SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
	//api.SetVariable("tessedit_char_blacklist", "xyz");
	//api.SetVariable("preserve_interword_spaces", "1");

	//const std::string image_filepath("../data/language_processing/phototest.tif");
	//const std::string image_filepath("../data/language_processing/eurotext.tif");
	//const std::string image_filepath("../data/language_processing/receipt.png");
	//const std::string image_filepath("../data/language_processing/road-sign-1.jpg");
	//const std::string image_filepath("../data/language_processing/road-sign-2-768x347.jpg");
	//const std::string image_filepath("../data/language_processing/road-sign-3-300x112.jpg");
	//const std::string image_filepath("../data/language_processing/korean_article_1.png");
	//const std::string image_filepath("../data/language_processing/korean_article_2.png");
	//const std::string image_filepath("../data/language_processing/korean_newspaper_1.png");
	const std::string image_filepath("../data/language_processing/korean_newspaper_2.png");
	//const std::string image_filepath("../data/language_processing/korean_newspaper_3.png");
	//const std::string image_filepath("../data/language_processing/korean_newspaper_4.png");

#if false
	// Open input image with leptonica library.
	Pix *image = pixRead(image_filepath.c_str());
	if (!image)
	{
		std::cerr << "Failed to load an image: " << image_filepath << std::endl;
		return;
	}

	api.SetImage(image);
#else
	// Open input image with OpenCV library.
	const cv::Mat img(cv::imread(image_filepath, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cerr << "Failed to load an image: " << image_filepath << std::endl;
		return;
	}

	api.SetImage(img.data, img.cols, img.rows, img.channels(), img.step);
#endif
	//api.SetRectangle(left, top, width, height);

	// Get OCR result.
	std::unique_ptr<char[]> outText(api.GetUTF8Text());
	if (outText.get())
	{
#if false
		std::ofstream stream("./tess_ocr_results.txt");
		if (stream.is_open())
			stream << outText.get() << std::endl;
#else
		std::wcout.imbue(std::locale("kor"));
		//std::wcin.imbue(std::locale("kor"));

		std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

		//std::cout << "OCR output:\n" << outText.get() << std::endl;  // Korean is not displayed correctly.
		std::wcout << L"OCR output:\n" << conv.from_bytes(outText.get()) << std::endl;
#endif
	}

	// Destroy used object and release memory.
	api.End();

#if false
	pixDestroy(&image);
#endif
}

// REF [site] >> https://github.com/tesseract-ocr/tesseract/wiki/APIExample
void text_line_recognition_example()
{
	const std::string tessdata_dir_path("../data/language_processing/tessdata/");
	const std::string tesslang("eng");

	tesseract::TessBaseAPI api;
	//if (api.Init(nullptr, tesslang.c_str(), tesseract::OEM_DEFAULT))
	if (api.Init(tessdata_dir_path.c_str(), tesslang.c_str(), tesseract::OEM_DEFAULT))
	{
		std::cerr << "Failed to initialize tesseract." << std::endl;
		return;
	}

	const std::string image_filepath("../data/language_processing/phototest.tif");

	Pix *image = pixRead(image_filepath.c_str());
	if (!image)
	{
		std::cerr << "Failed to load an image: " << image_filepath << std::endl;
		return;
	}

	api.SetImage(image);

	std::cout << "Start processing..." << std::endl;
	const auto start_time = std::chrono::high_resolution_clock::now();
	std::unique_ptr<Boxa> boxes(api.GetComponentImages(tesseract::RIL_TEXTLINE, true, NULL, NULL));
	std::cout << "End processing: " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
	std::cout << "Found " << boxes->n << " textline image components." << std::endl;

	for (int i = 0; i < boxes->n; ++i)
	{
		std::unique_ptr<BOX> box(boxaGetBox(boxes.get(), i, L_CLONE));
		api.SetRectangle(box->x, box->y, box->w, box->h);
		const int confidence = api.MeanTextConf();  // [0, 100].
		std::cout << "Box[" << i << "]: x=" << box->x << ", y=" << box->y << ", w=" << box->w << ", h=" << box->h << ", confidence: " << confidence << std::endl;

		std::unique_ptr<char[]> ocrResult(api.GetUTF8Text());
		if (ocrResult.get())
		{
			std::wcout.imbue(std::locale("kor"));
			//std::wcin.imbue(std::locale("kor"));

			std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

			std::wcout << L"Text: " << conv.from_bytes(ocrResult.get()) << std::endl;
		}
		else
			std::cout << "No text." << std::endl;
	}

	// Destroy used object and release memory.
	api.End();
	pixDestroy(&image);
}

// REF [site] >> https://github.com/tesseract-ocr/tesseract/wiki/APIExample
void result_iterator_example()
{
	const std::string tessdata_dir_path("../data/language_processing/tessdata/");
	const std::string tesslang("eng");

	tesseract::TessBaseAPI api;
	//if (api.Init(nullptr, tesslang.c_str(), tesseract::OEM_DEFAULT))
	if (api.Init(tessdata_dir_path.c_str(), tesslang.c_str(), tesseract::OEM_DEFAULT))
	{
		std::cerr << "Failed to initialize tesseract." << std::endl;
		return;
	}

	const std::string image_filepath("../data/language_processing/phototest.tif");

	Pix *image = pixRead(image_filepath.c_str());
	if (!image)
	{
		std::cerr << "Failed to load an image: " << image_filepath << std::endl;
		return;
	}

	api.SetImage(image);
	if (api.Recognize(nullptr) < 0)
	{
		std::cerr << "Failed to recognize: " << image_filepath << std::endl;
		return;
	}

	tesseract::ResultIterator *ri = api.GetIterator();
	if (ri)
	{
		const tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
		do
		{
			const float confidence = ri->Confidence(level);
			int x1, y1, x2, y2;
			ri->BoundingBox(level, &x1, &y1, &x2, &y2);
			std::cout << "Confidence: " << confidence << "; BoundingBox: " << x1 << ", " << y1 << ", " << x2 << ", " << y2 << std::endl;

			std::unique_ptr<char[]> word(ri->GetUTF8Text(level));
			if (word.get())
			{
				std::wcout.imbue(std::locale("kor"));
				//std::wcin.imbue(std::locale("kor"));

				std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

				std::wcout << L"Word: " << conv.from_bytes(word.get()) << std::endl;
			}
			else
				std::cout << "No word." << std::endl;
		} while (ri->Next(level));
	}

	// Destroy used object and release memory.
	api.End();
	pixDestroy(&image);
}

// REF [site] >> https://github.com/tesseract-ocr/tesseract/wiki/APIExample
void orientation_and_script_detection_example()
{
	const std::string tessdata_dir_path("../data/language_processing/tessdata/");
	const std::string tesslang("eng");

	tesseract::TessBaseAPI api;
	//if (api.Init(nullptr, tesslang.c_str(), tesseract::OEM_DEFAULT))
	if (api.Init(tessdata_dir_path.c_str(), tesslang.c_str(), tesseract::OEM_DEFAULT))
	{
		std::cerr << "Failed to initialize tesseract." << std::endl;
		return;
	}
	api.SetPageSegMode(tesseract::PSM_AUTO_OSD);  // Orientation and script detection (OSD).

	const std::string image_filepath("../data/language_processing/eurotext.tif");

	Pix *image = pixRead(image_filepath.c_str());
	if (!image)
	{
		std::cerr << "Failed to load an image: " << image_filepath << std::endl;
		return;
	}

	api.SetImage(image);
	if (api.Recognize(nullptr) < 0)
	{
		std::cerr << "Failed to recognize: " << image_filepath << std::endl;
		return;
	}

	tesseract::PageIterator *it = api.AnalyseLayout();
	if (it)
	{
		tesseract::Orientation orientation;
		tesseract::WritingDirection direction;
		tesseract::TextlineOrder order;
		float deskew_angle;

		//do
		//{
			it->Orientation(&orientation, &direction, &order, &deskew_angle);

			std::cout << "Orientation: " << orientation << std::endl;
			std::cout << "WritingDirection: " << direction << std::endl;
			std::cout << "TextlineOrder : " << order << std::endl;
			std::cout << "Deskew angle: " << deskew_angle << std::endl;
		//} while (it->Next(tesseract::RIL_TEXTLINE));
	}

	// Destroy used object and release memory.
	api.End();
	pixDestroy(&image);
}

// REF [site] >> https://github.com/tesseract-ocr/tesseract/wiki/APIExample
void iterator_over_the_classifier_choices_for_a_single_symbol()
{
	const std::string tessdata_dir_path("../data/language_processing/tessdata/");
	const std::string tesslang("eng");

	tesseract::TessBaseAPI api;
	//if (api.Init(nullptr, tesslang.c_str(), tesseract::OEM_DEFAULT))
	if (api.Init(tessdata_dir_path.c_str(), tesslang.c_str(), tesseract::OEM_DEFAULT))
	{
		std::cerr << "Failed to initialize tesseract." << std::endl;
		return;
	}

	const std::string image_filepath("../data/language_processing/phototest.tif");
	Pix *image = pixRead(image_filepath.c_str());
	if (!image)
	{
		std::cerr << "Failed to load an image: " << image_filepath << std::endl;
		return;
	}

	api.SetVariable("save_blob_choices", "T");

	api.SetImage(image);
	api.SetRectangle(37, 228, 548, 31);
	if (api.Recognize(nullptr) < 0)
	{
		std::cerr << "Failed to recognize: " << image_filepath << std::endl;
		return;
	}

	tesseract::ResultIterator *ri = api.GetIterator();
	if (ri)
	{
		const tesseract::PageIteratorLevel level = tesseract::RIL_SYMBOL;
		do
		{
			const char *symbol = ri->GetUTF8Text(level);
			//std::unique_ptr<char[]> symbol(ri->GetUTF8Text(level));
			const float confidence = ri->Confidence(level);
			if (nullptr != symbol)
			{
				std::cout << "Symbol: " << symbol << ", confidence: " << confidence;
				bool indent = false;
				tesseract::ChoiceIterator ci(*ri);
				do
				{
					if (indent) std::cout << "\t\t ";
					std::cout << "\t- ";
					const char *choice = ci.GetUTF8Text();
					//std::unique_ptr<const char[]> choice(ci.GetUTF8Text());
					if (choice)
						std::cout << choice << ", confidence: " << ci.Confidence() << std::endl;
					indent = true;
				} while (ci.Next());
			}
			std::cout << "---------------------------------------------" << std::endl;
		} while ((ri->Next(level)));
	}

	// Destroy used object and release memory.
	api.End();
	pixDestroy(&image);
}

// REF [site] >> https://github.com/tesseract-ocr/tesseract/wiki/APIExample
void creating_searchable_pdf_from_image()
{
	const std::string tessdata_dir_path("../data/language_processing/tessdata/");
	const std::string tesslang("eng");

	tesseract::TessBaseAPI api;
	//if (api.Init(nullptr, tesslang.c_str(), tesseract::OEM_DEFAULT))
	if (api.Init(tessdata_dir_path.c_str(), tesslang.c_str(), tesseract::OEM_DEFAULT))
	{
		std::cerr << "Failed to initialize tesseract." << std::endl;
		return;
	}

	const std::string image_filepath("../data/language_processing/phototest.tif");

	const std::string output_base("my_first_tesseract_pdf");
	const int timeout_ms = 5000;
	const char *retry_config = nullptr;
	const bool textonly = false;
	const int jpg_quality = 92;

	//tesseract::TessPDFRenderer renderer(output_base.c_str(), api.GetDatapath(), textonly, jpg_quality);
	tesseract::TessPDFRenderer renderer(output_base.c_str(), api.GetDatapath(), textonly);

	std::cout << "Start processing..." << std::endl;
	const auto start_time = std::chrono::high_resolution_clock::now();
	const bool succeed = api.ProcessPages(image_filepath.c_str(), retry_config, timeout_ms, &renderer);  // Needs "tessdata/pdf.ttf".
	std::cout << "End processing: " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;

	if (!succeed)
		std::cerr << "Error during processing." << std::endl;

	api.End();
}

#if defined(__linux) || defined(__linux__) || defined(linux) || defined(__unix) || defined(__unix__) || defined(unix)
void monitorProgress(ETEXT_DESC *monitor, int page)
{
	while (true)
	{
		std::cout << '\r' << monitor[page].progress << '%%' << std::flush;
		if (100 == monitor[page].progress)
			break;
	}
}

void ocrProcess(tesseract::TessBaseAPI *api, ETEXT_DESC *monitor)
{
	api->Recognize(monitor);
}

// REF [site] >> https://github.com/tesseract-ocr/tesseract/wiki/APIExample
void monitoring_ocr_progress()
{
	const std::string tessdata_dir_path("../data/language_processing/tessdata/");
	const std::string tesslang("eng");

	tesseract::TessBaseAPI api;
	//if (api.Init(nullptr, tesslang.c_str(), tesseract::OEM_DEFAULT))
	if (api.Init(tessdata_dir_path.c_str(), tesslang.c_str(), tesseract::OEM_DEFAULT))
	{
		std::cerr << "Failed to initialize tesseract." << std::endl;
		return;
	}

	api.SetPageSegMode(tesseract::PSM_AUTO);

	const std::string image_filepath("../data/language_processing/phototest.tif");
	Pix *image = pixRead(image_filepath.c_str());
	if (!image)
	{
		std::cerr << "Failed to load an image: " << image_filepath << std::endl;
		return;
	}

	api.SetImage(image);

	ETEXT_DESC *monitor = new ETEXT_DESC();
	int page = 0;
	std::thread t1(ocrProcess, &api, monitor);
	std::thread t2(monitorProgress, monitor, page);
	t1.join();
	t2.join();

	pixDestroy(&image);

	std::unique_ptr<char[]> outText(api.GetUTF8Text());
	if (outText.get())
	{
		std::wcout.imbue(std::locale("kor"));
		//std::wcin.imbue(std::locale("kor"));

		std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

		std::wcout << L"Text: " << conv.from_bytes(outText.get()) << std::endl;
	}
	else
		std::cout << "No text." << std::endl;

	api.End();
}
#endif

void layout_analysis_example_1()
{
	const std::string tessdata_dir_path("../data/language_processing/tessdata/");
	const std::string tesslang("eng");

	tesseract::TessBaseAPI api;
	api.InitForAnalysePage();
	api.SetPageSegMode(tesseract::PSM_AUTO);

	//const std::string image_filepath("../data/language_processing/document_1.tiff");  // Multiple pages.
	//const std::string image_filepath("../data/language_processing/document_2.png");
	//const std::string image_filepath("../data/language_processing/document_3.png");
	const std::string image_filepath("../data/language_processing/document_4.png");
	//const std::string image_filepath("D:/work_biz/silicon_minds/DataAnalysis_bitbucket/app/form_text_recognition/receipt/epapyrus_20190618/receipt_1/img04.jpg");
	//const std::string image_filepath("D:/work_biz/silicon_minds/DataAnalysis_bitbucket/app/form_text_recognition/receipt/epapyrus_20190618/receipt_1_1/img04_01.png");

	Pix *image = pixRead(image_filepath.c_str());
	if (!image)
	{
		std::cerr << "Failed to load an image: " << image_filepath << std::endl;
		return;
	}

	api.SetImage(image);
	/*
	if (api.Recognize(nullptr) < 0)
	{
		std::cerr << "Failed to recognize: " << image_filepath << std::endl;
		return;
	}
	*/

	std::list<std::array<int, 4>> rectangles;

	std::cout << "Start analyzing layout..." << std::endl;
	const auto start_time = std::chrono::high_resolution_clock::now();
	tesseract::PageIterator *it = api.AnalyseLayout();
	if (it)
	{
		// Instead of RIL_WORD, you can use any other PageSegMode.
		const tesseract::PageIteratorLevel level = tesseract::RIL_PARA;

		int left, top, right, bottom;
		do
		{
			it->BoundingBox(level, &left, &top, &right, &bottom);
			rectangles.push_back({ left, top, right, bottom });
		} while (it->Next(level));
	}
	std::cout << "End analyzing layout: " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
	std::cout << "Found " << rectangles.size() << " regions." << std::endl;

	// Destroy used object and release memory.
	api.End();
	pixDestroy(&image);

	//--------------------
	cv::Mat img(cv::imread(image_filepath));
	for (const auto &rect : rectangles)
	{
		//std::cout << "BoundingBox: " << rect[0] << ", " << rect[1] << ", " << rect[2] << ", " << rect[3] << std::endl;
		cv::rectangle(img, cv::Point(rect[0], rect[1]), cv::Point(rect[2], rect[3]), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
		cv::imshow("Layout Analysis", img);
		cv::waitKey(0);
	}

	cv::destroyAllWindows();
}

// REF [function] >> TessBaseAPI::GetTSVText() in https://github.com/tesseract-ocr/tesseract/blob/master/src/api/baseapi.cpp
void layout_analysis_example_2()
{
	const std::string tessdata_dir_path("../data/language_processing/tessdata/");
	const std::string tesslang("eng");

	tesseract::TessBaseAPI api;
	//if (api.Init(nullptr, tesslang.c_str(), tesseract::OEM_DEFAULT))
	if (api.Init(tessdata_dir_path.c_str(), tesslang.c_str(), tesseract::OEM_DEFAULT))
	{
		std::cerr << "Failed to initialize tesseract." << std::endl;
		return;
	}
	api.SetPageSegMode(tesseract::PSM_AUTO);

	//const std::string image_filepath("../data/language_processing/document_1.tiff");  // Multiple pages.
	//const std::string image_filepath("../data/language_processing/document_2.png");
	//const std::string image_filepath("../data/language_processing/document_3.png");
	const std::string image_filepath("../data/language_processing/document_4.png");
	//const std::string image_filepath("D:/work_biz/silicon_minds/DataAnalysis_bitbucket/app/form_text_recognition/receipt/epapyrus_20190618/receipt_1/img04.jpg");
	//const std::string image_filepath("D:/work_biz/silicon_minds/DataAnalysis_bitbucket/app/form_text_recognition/receipt/epapyrus_20190618/receipt_1_1/img04_01.png");

	Pix *image = pixRead(image_filepath.c_str());
	if (!image)
	{
		std::cerr << "Failed to load an image: " << image_filepath << std::endl;
		return;
	}

	api.SetImage(image);
	if (api.Recognize(nullptr) < 0)
	{
		std::cerr << "Failed to recognize: " << image_filepath << std::endl;
		return;
	}

	std::list<std::array<int, 4>> rectangles;

	std::cout << "Start analyzing layout..." << std::endl;
	const auto start_time = std::chrono::high_resolution_clock::now();
	tesseract::ResultIterator *ri = api.GetIterator();
	if (ri)
	{
		const tesseract::PageIteratorLevel level = tesseract::RIL_PARA;

		int left, top, right, bottom;
		do
		{
			ri->BoundingBox(level, &left, &top, &right, &bottom);
			rectangles.push_back({ left, top, right, bottom });
		} while (ri->Next(level));
	}
	std::cout << "End analyzing layout: " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
	std::cout << "Found " << rectangles.size() << " regions." << std::endl;

	// Destroy used object and release memory.
	api.End();
	pixDestroy(&image);

	//--------------------
	cv::Mat img(cv::imread(image_filepath));
	for (const auto &rect : rectangles)
	{
		//std::cout << "BoundingBox: " << rect[0] << ", " << rect[1] << ", " << rect[2] << ", " << rect[3] << std::endl;
		cv::rectangle(img, cv::Point(rect[0], rect[1]), cv::Point(rect[2], rect[3]), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
		cv::imshow("Layout Analysis", img);
		cv::waitKey(0);
	}

	cv::destroyAllWindows();
}

void binarization_example()
{
	const std::string tessdata_dir_path("../data/language_processing/tessdata/");
	const std::string tesslang("eng");

	tesseract::TessBaseAPI api;
	//if (api.Init(nullptr, tesslang.c_str(), tesseract::OEM_DEFAULT))
	if (api.Init(tessdata_dir_path.c_str(), tesslang.c_str(), tesseract::OEM_DEFAULT))
	{
		std::cerr << "Failed to initialize tesseract." << std::endl;
		return;
	}

	//const std::string image_filepath("../data/language_processing/phototest.tif");
	//const std::string image_filepath("../data/language_processing/eurotext.tif");
	//const std::string image_filepath("../data/language_processing/receipt.png");
	//const std::string image_filepath("../data/language_processing/road-sign-1.jpg");
	//const std::string image_filepath("../data/language_processing/road-sign-2-768x347.jpg");
	//const std::string image_filepath("../data/language_processing/road-sign-3-300x112.jpg");
	//const std::string image_filepath("../data/language_processing/korean_article_1.png");
	//const std::string image_filepath("../data/language_processing/korean_article_2.png");
	//const std::string image_filepath("../data/language_processing/korean_newspaper_1.png");
	//const std::string image_filepath("../data/language_processing/korean_newspaper_2.png");
	//const std::string image_filepath("../data/language_processing/korean_newspaper_3.png");
	//const std::string image_filepath("../data/language_processing/korean_newspaper_4.png");
	const std::string image_filepath("D:/work/SWL_github/python/test/language_processing/receipt/receipt_text_line_en/0-000003.jpg");

	Pix *image = pixRead(image_filepath.c_str());
	if (!image)
	{
		std::cerr << "Failed to load an image: " << image_filepath << std::endl;
		return;
	}

	api.SetImage(image);

	std::cout << "Start binarizing..." << std::endl;
	const auto start_time = std::chrono::high_resolution_clock::now();
	Pix *pix1 = api.GetThresholdedImage();
	std::cout << "End binarizing: " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;

	if (true)
	{
		// Pix (leptonica) -> cv::Mat (OpenCV).
		Pix *pix8 = pixConvert1To8(NULL, pix1, 255, 0);
		const cv::Mat bw(my_leptonica::pix8ToMat(pix8));
		const cv::Mat rgba(my_leptonica::pix32ToMat(image));

		cv::imshow("Image", rgba);
		cv::imshow("Binarization", bw);
		cv::waitKey(0);

		cv::destroyAllWindows();

		pixDestroy(&pix8);
	}
	
	// Destroy used object and release memory.
	api.End();
	pixDestroy(&image);
	pixDestroy(&pix1);
}

}  // namespace local
}  // unnamed namespace

namespace my_tesseract {

}  // namespace my_tesseract

int tesseract_main(int argc, char *argv[])
{
#if true
	//local::simple_text_recognition_example();

	//local::text_recognition_example();
	//local::text_line_recognition_example();  // Results are not good.
	//local::result_iterator_example();
	//local::orientation_and_script_detection_example();
	//local::iterator_over_the_classifier_choices_for_a_single_symbol();
	//local::creating_searchable_pdf_from_image();
#if defined(__linux) || defined(__linux__) || defined(linux) || defined(__unix) || defined(__unix__) || defined(unix)
	local::monitoring_ocr_progress();
#endif

	//local::layout_analysis_example_1();
	//local::layout_analysis_example_2();

	local::binarization_example();

	return 0;
#else
	std::vector<char *> args = {
		argv[0],
		"--tessdata-dir", "../data/language_processing/tessdata/",
		"-l", "kor+eng",
		"--oem", "3",
		"--psm", "3",
		//"../data/language_processing/phototest.tif",
		"../data/language_processing/korean_article_2.png",
		//"tess_ocr_results",
		"stdout",
	};

	// REF [file] >> ${TESSERACT_HOME}/src/api/tesseractmain.cpp
	return local::tesseract_main((int)args.size(), &args[0]);
#endif
}
