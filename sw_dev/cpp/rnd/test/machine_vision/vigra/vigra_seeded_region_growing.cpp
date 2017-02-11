#include <vigra/seededregiongrowing.hxx>
#include <vigra/distancetransform.hxx>
#include <vigra/pixelneighborhood.hxx>
#include <vigra/multi_array.hxx>
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <list>
#include <cstdlib>


namespace {
namespace local {

// REF [site] >> ${VIGRA_HOME}/doc/vigra/group__SeededRegionGrowing.html
void seeded_region_growing_example()
{
	const int width = 400, height = 300;

	vigra::MultiArray<2, int> points(height, width);
	vigra::MultiArray<2, float> dist(height, width);

	const int max_region_label = 100;

	// Throw in some random points.
	for (int i = 1; i <= max_region_label; ++i)
		points(height * std::rand() / RAND_MAX , width * std::rand() / RAND_MAX) = i;

	// Calculate Euclidean distance transform.
	vigra::distanceTransform(points, dist, 0, 2);

	// Init statistics functor.
	vigra::ArrayOfRegionStatistics<vigra::SeedRgDirectValueFunctor<float> > stats(max_region_label);

	// Find Voronoi region of each point (the point image is overwritten with the Voronoi region labels).
	{
		boost::timer::auto_cpu_timer timer;
#if 0
		vigra::seededRegionGrowing(dist, points, points, stats, vigra::CompleteGrow, vigra::FourNeighborCode(), vigra::NumericTraits<double>::max());
#else
		vigra::fastSeededRegionGrowing(dist, points, stats, vigra::CompleteGrow, vigra::FourNeighborCode(), vigra::NumericTraits<double>::max());
#endif
	}

	// Display.
	cv::Mat input_img(height, width, CV_32FC1, cv::Scalar::all(0)), label_img(height, width, CV_32SC1, cv::Scalar::all(0));
	for (int i = 0; i < input_img.rows; ++i)
		for (int j = 0; j < input_img.cols; ++j)
		{
			input_img.at<float>(i, j) = dist(i, j);
			label_img.at<int>(i, j) = points(i, j);
		}

    double minVal = 0.0, maxVal = 0.0;
	cv::minMaxLoc(input_img, &minVal, &maxVal);
	input_img.convertTo(input_img, CV_32FC1, 1.0 / maxVal, 0.0);
	cv::minMaxLoc(label_img, &minVal, &maxVal);
	label_img.convertTo(label_img, CV_32FC1, 1.0 / maxVal, 0.0);

	cv::imshow("Seeded region growing - input", input_img);
	cv::imshow("Seeded region growing - label", label_img);

	cv::waitKey(0);

	cv::destroyAllWindows();
}

void seeded_region_growing()
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
		std::cout << "Image file not found: " << input_filename << std::endl;
		return;
	}

    double minVal = 0.0, maxVal = 0.0;
	cv::minMaxLoc(input_img, &minVal, &maxVal);
	input_img.convertTo(input_img, CV_32FC1, 1.0 / maxVal, 0.0);

	const int width = input_img.cols, height = input_img.rows;

	vigra::MultiArray<2, int> points(height, width);
	points.init(0);
	vigra::MultiArray<2, float> dist(height, width);

	for (int i = 0; i < input_img.rows; ++i)
		for (int j = 0; j < input_img.cols; ++j)
			dist(i, j) = input_img.at<float>(i, j);

	const int max_region_label = (int)seed_points.size();
	int idx = 1;
	for (std::list<cv::Point>::const_iterator cit = seed_points.begin(); cit != seed_points.end(); ++cit)
		points(cit->y, cit->x) = idx++;

	// Init statistics functor.
	vigra::ArrayOfRegionStatistics<vigra::SeedRgDirectValueFunctor<float> > stats(max_region_label);

	// Find voronoi region of each point (the point image is overwritten with the voronoi region labels).
	{
		boost::timer::auto_cpu_timer timer;
#if 1
		vigra::seededRegionGrowing(dist, points, points, stats, vigra::CompleteGrow, vigra::FourNeighborCode(), vigra::NumericTraits<double>::max());
#else
		vigra::fastSeededRegionGrowing(dist, points, stats, vigra::CompleteGrow, vigra::FourNeighborCode(), vigra::NumericTraits<double>::max());
#endif
	}

	// Display.
	cv::Mat label_img(height, width, CV_32SC1, cv::Scalar::all(0));
	for (int i = 0; i < label_img.rows; ++i)
		for (int j = 0; j < label_img.cols; ++j)
			label_img.at<int>(i, j) = points(i, j);

	cv::minMaxLoc(label_img, &minVal, &maxVal);
	label_img.convertTo(label_img, CV_32FC1, 1.0 / maxVal, 0.0);

	cv::imshow("Seeded region growing - input", input_img);
	cv::imshow("seeded region growing - label", label_img);

	cv::waitKey(0);

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_vigra {

void seeded_region_growing()
{
	// Seeded region growing (SRG).
	//	- REF [paper] >> "Seeded Region Growing", R. Adams and L. Bischof, TPAMI 1994.

	//local::seeded_region_growing_example();
	local::seeded_region_growing();
}

}  // namespace my_vigra
