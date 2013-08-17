#include <vigra/multi_watersheds.hxx>
#include <vigra/convolution.hxx>
#include <vigra/combineimages.hxx>
#include <vigra/multi_array.hxx>
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <list>


namespace {
namespace local {

// [ref] ${VIGRA_HOME}/doc/vigra/group__SeededRegionGrowing.html
void watershed_region_growing_example()
{
#if 0
	const std::string input_filename("./data/segmentation/beach.png");
	const std::string output_filename("./data/segmentation/beach_segmented.png");

	std::list<cv::Point> seed_points;  // (x, y) = (col, row).
	seed_points.push_back(cv::Point(97, 179));
	seed_points.push_back(cv::Point(87, 25));
	seed_points.push_back(cv::Point(120, 84));
	seed_points.push_back(cv::Point(184, 130));
	seed_points.push_back(cv::Point(49, 232));
#elif 0
	const std::string input_filename("./data/segmentation/grain.png");
	const std::string output_filename("./data/segmentation/grain_segmented.png");

	std::list<cv::Point> seed_points;  // (x, y) = (col, row).
	seed_points.push_back(cv::Point(135, 90));
	seed_points.push_back(cv::Point(155, 34));
	seed_points.push_back(cv::Point(83, 140));
	seed_points.push_back(cv::Point(238, 25));
	seed_points.push_back(cv::Point(19, 41));
	seed_points.push_back(cv::Point(14, 166));
	seed_points.push_back(cv::Point(88, 189));
	seed_points.push_back(cv::Point(291, 64));
#elif 1
	const std::string input_filename("./data/segmentation/brain_small.png");
	const std::string output_filename("./data/segmentation/brain_small_segmented.png");

	std::list<cv::Point> seed_points;  // (x, y) = (col, row).
	seed_points.push_back(cv::Point(236, 157));
	seed_points.push_back(cv::Point(284, 310));
	seed_points.push_back(cv::Point(45, 274));
#endif

	cv::Mat input_img = cv::imread(input_filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (input_img.empty())
	{
		std::cout << "image file not found: " << input_filename << std::endl;
		return;
	}

    double minVal = 0.0, maxVal = 0.0;
	cv::minMaxLoc(input_img, &minVal, &maxVal);
	input_img.convertTo(input_img, CV_32FC1, 1.0 / maxVal, 0.0);

	vigra::MultiArray<2, float> src(vigra::Shape2(input_img.rows, input_img.cols));
	for (int i = 0; i < input_img.rows; ++i)
		for (int j = 0; j < input_img.cols; ++j)
			src(i, j) = input_img.at<float>(i, j);

	// compute gradient magnitude at scale 3.0 as a boundary indicator.
    vigra::MultiArray<2, float> gradMag(src.shape());
	const double scale = 3.0;
#if 1
	vigra::gaussianGradientMagnitude(vigra::srcImageRange(src), vigra::destImage(gradMag), scale);
#else
    vigra::MultiArray<2, float> gradx(src.shape()), grady(src.shape());
	vigra::gaussianGradient(src, gradx, grady, scale);
	vigra::combineTwoImages(gradx, grady, gradMag, vigra::MagnitudeFunctor<float>());
#endif

	vigra::MultiArray<2, unsigned int> labeling(src.shape());
#if 0
	// example 1
    {
		// the pixel type of the destination image must be large enough to hold numbers up to 'max_region_label' to prevent overflow.

		boost::timer::auto_cpu_timer timer;
        // call watershed algorithm for 4-neighborhood, leave a 1-pixel boundary between regions,
        // and autogenerate seeds from all gradient minima where the magnitude is below 0.0005.
		const unsigned int max_region_label = vigra::watershedsRegionGrowing(
			gradMag, labeling,
            vigra::FourNeighborCode(),
            vigra::WatershedOptions().keepContours().seedOptions(vigra::SeedOptions().minima().threshold(0.0005))
		);
    }
	const int display_method = 1;
#elif 1
	// example 2
	{
		// compute seeds beforehand (use connected components of all pixels where the gradient is below 0.01).
		const unsigned int max_region_label = vigra::generateWatershedSeeds(
			gradMag, labeling,
			vigra::IndirectNeighborhood,
			vigra::SeedOptions().levelSets(0.01)
		);

		// quantize the gradient image to 256 gray levels.
		vigra::MultiArray<2, unsigned char> gradMag256(src.shape());
		vigra::FindMinMax<float> minmax; 
		vigra::inspectImage(gradMag, minmax);  // find original range.
		vigra::transformImage(gradMag, gradMag256, vigra::linearRangeMapping(minmax, 0, 255));

		boost::timer::auto_cpu_timer timer;
		// call the turbo algorithm with 256 bins, using 8-neighborhood.
		vigra::watershedsRegionGrowing(
			gradMag256, labeling,
			vigra::WatershedOptions().turboAlgorithm(256)
		);
	}
	const int display_method = 0;
#elif 0
	// example 3
	{
		// get seeds from somewhere, e.g. an interactive labeling program,
		// make sure that label 1 corresponds to the background.
		for (std::list<cv::Point>::const_iterator cit = seed_points.begin(); cit != seed_points.end(); ++cit)
			labeling(cit->y, cit->x) = 1;

		boost::timer::auto_cpu_timer timer;
		// bias the watershed algorithm so that the background is preferred by reducing the cost for label 1 to 90%.
		const unsigned int max_region_label = vigra::watershedsRegionGrowing(
			gradMag, labeling,
			vigra::WatershedOptions().biasLabel(1, 0.9)
		);
	}
	const int display_method = 2;
#endif

	// display.
	cv::Mat label_img(input_img.size(), CV_32SC1, cv::Scalar::all(0));
	for (int i = 0; i < label_img.rows; ++i)
		for (int j = 0; j < label_img.cols; ++j)
			label_img.at<int>(i, j) = labeling(i, j);

	switch (display_method)
	{
	case 0:
		cv::minMaxLoc(label_img, &minVal, &maxVal);
		label_img.convertTo(label_img, CV_32FC1, 1.0 / maxVal, 0.0);
		break;
	case 1:
		label_img = 0 == label_img;
		break;
	case 2:
		label_img = 1 == label_img;
		break;
	}

	cv::imshow("watershed region growing - input", input_img);
	cv::imshow("watershed region growing - label", label_img);

	cv::waitKey(0);

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_vigra {

void watershed_region_growing()
{
	local::watershed_region_growing_example();
}

}  // namespace my_vigra
