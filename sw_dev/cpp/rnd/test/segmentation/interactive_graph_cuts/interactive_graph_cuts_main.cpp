#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/graphicalmodel/space/grid_space.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#include <opengm/inference/auxiliary/minstcutboost.hxx>
#include <opengm/functions/function_registration.hxx>
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <string>
#include <iostream>
#include <cmath>
#include <cassert>


//#define __USE_GRID_SPACE 1
#define __USE_8_NEIGHBORHOOD_SYSTEM 1

namespace {
namespace local {

#if defined(__USE_GRID_SPACE)
typedef opengm::GridSpace<std::size_t, std::size_t> Space;
#else
typedef opengm::SimpleDiscreteSpace<std::size_t, std::size_t> Space;
#endif
typedef opengm::ExplicitFunction<double> ExplicitFunction;

// construct a graphical model with
// - addition as the operation (template parameter Adder)
typedef opengm::GraphicalModel<double, opengm::Adder, ExplicitFunction, Space> GraphicalModel;

// this function maps a node (x, y) in the grid to a unique variable index
inline std::size_t getVariableIndex(const std::size_t Nx, const std::size_t x, const std::size_t y)
{
	return x + Nx * y;
}

// [ref]
//	createGraphicalModelForPottsModel() in ${CPP_RND_HOME}/test/probabilistic_graphical_model/opengm/opengm_inference_algorithms.cpp
//	${OPENGM_HOME}/src/examples/image-processing-examples/grid_potts.cxx
//	"Interactive Graph Cuts for Optimal Boundary & Region Segmentation of Objects in N-D Images", Yuri Y. Boykov and Marie-Pierre Jolly, ICCV, 2001.
bool create_graphical_model(const cv::Mat &img, const std::size_t numOfLabels, const double lambda, const double sigma, const cv::Mat &histForeground, const cv::Mat &histBackground, GraphicalModel &gm)
{
	// model parameters (global variables are used only in example code)
	const std::size_t Nx = img.cols;  // width of the grid
	const std::size_t Ny = img.rows;  // height of the grid

	// construct a label space with
	//	- Nx * Ny variables
	//	- each having numOfLabels many labels
#if defined(__USE_GRID_SPACE)
	Space space(Nx, Ny, numOfLabels);
#else
	Space space(Nx * Ny, numOfLabels);
#endif

	gm = GraphicalModel(space);

	const double sigma2 = 2.0 * sigma * sigma;
	const double tol = 1.0e-50;
	//const double minVal = std::numeric_limits<double>::min();
	const double minVal = tol;
	const std::size_t shape1[] = { numOfLabels };
	const std::size_t shape2[] = { numOfLabels, numOfLabels };
	for (std::size_t y = 0; y < Ny; ++y)
	{
		for (std::size_t x = 0; x < Nx; ++x)
		{
			// Add 1st order functions and factors.
			// For each node (x, y) in the grid, i.e. for each variable variableIndex(Nx, x, y) of the model,
			// add one 1st order functions and one 1st order factor
			{
				const unsigned char pix = img.at<unsigned char>(y, x);
				const double probForeground = (double)histForeground.at<float>(pix);
				assert(0.0 <= probForeground && probForeground <= 1.0);
				const double probBackground = (double)histBackground.at<float>(pix);
				assert(0.0 <= probBackground && probBackground <= 1.0);

				// function
				ExplicitFunction func1(shape1, shape1 + 1);
				func1(0) = -lambda * std::log(std::fabs(probBackground) > tol ? probBackground : minVal);  // background (state = 1)
				func1(1) = -lambda * std::log(std::fabs(probForeground) > tol ? probForeground : minVal);  // foreground (state = 0)

				const GraphicalModel::FunctionIdentifier fid1 = gm.addFunction(func1);

				// factor
				const std::size_t variableIndices[] = { getVariableIndex(Nx, x, y) };
				gm.addFactor(fid1, variableIndices, variableIndices + 1);
			}

			// Add 2nd order functions and factors.
			// For each pair of nodes (x1, y1), (x2, y2) which are adjacent on the grid,
			// add one factor that connects the corresponding variable indices.
			// An 4-neighborhood or 8-neighborhood system in 2D (4-connectivity or 8-connectivity).
			{
				// factor
				const unsigned char pix1 = img.at<unsigned char>(y, x);
				if (x + 1 < Nx)  // (x, y) -- (x + 1, y)
				{
					const unsigned char pix2 = img.at<unsigned char>(y, x + 1);
					const double dist = 1.0;
					const double B = std::exp(-double(pix2 - pix1) * double(pix2 - pix1) / sigma2) / dist;

					// function
					ExplicitFunction func2(shape2, shape2 + 2);
					func2(0, 0) = 0.0;
					func2(0, 1) = B;
					func2(1, 0) = B;
					func2(1, 1) = 0.0;
					const GraphicalModel::FunctionIdentifier fid2 = gm.addFunction(func2);

					std::size_t variableIndices[] = { getVariableIndex(Nx, x, y), getVariableIndex(Nx, x + 1, y) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				if (y + 1 < Ny)  // (x, y) -- (x, y + 1)
				{
					const unsigned char pix2 = img.at<unsigned char>(y + 1, x);
					const double dist = 1.0;
					const double B = std::exp(-double(pix2 - pix1) * double(pix2 - pix1) / sigma2) / dist;

					// function
					ExplicitFunction func2(shape2, shape2 + 2);
					func2(0, 0) = 0.0;
					func2(0, 1) = B;
					func2(1, 0) = B;
					func2(1, 1) = 0.0;
					const GraphicalModel::FunctionIdentifier fid2 = gm.addFunction(func2);

					std::size_t variableIndices[] = { getVariableIndex(Nx, x, y), getVariableIndex(Nx, x, y + 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
#if defined(__USE_8_NEIGHBORHOOD_SYSTEM)
				if (x + 1 < Nx && y + 1 < Ny)  // (x, y) -- (x + 1, y + 1)
				{
					const unsigned char pix2 = img.at<unsigned char>(y + 1, x + 1);
					const double dist = std::sqrt(2.0); //2.0;
					//const double dist = 1.0;
					const double B = std::exp(-double(pix2 - pix1) * double(pix2 - pix1) / sigma2) / dist;

					// function
					ExplicitFunction func2(shape2, shape2 + 2);
					func2(0, 0) = 0.0;
					func2(0, 1) = B;
					func2(1, 0) = B;
					func2(1, 1) = 0.0;
					const GraphicalModel::FunctionIdentifier fid2 = gm.addFunction(func2);

					std::size_t variableIndices[] = { getVariableIndex(Nx, x, y), getVariableIndex(Nx, x + 1, y + 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				if (x + 1 < Nx && int(y - 1) >= 0)  // (x, y) -- (x + 1, y - 1)
				{
					const unsigned char pix2 = img.at<unsigned char>(y - 1, x + 1);
					const double dist = std::sqrt(2.0); //2.0;
					//const double dist = 1.0;
					const double B = std::exp(-double(pix2 - pix1) * double(pix2 - pix1) / sigma2) / dist;

					// function
					ExplicitFunction func2(shape2, shape2 + 2);
					func2(0, 0) = 0.0;
					func2(0, 1) = B;
					func2(1, 0) = B;
					func2(1, 1) = 0.0;
					const GraphicalModel::FunctionIdentifier fid2 = gm.addFunction(func2);

					std::size_t variableIndices[] = { getVariableIndex(Nx, x, y), getVariableIndex(Nx, x + 1, y - 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
#endif  // __USE_8_NEIGHBORHOOD_SYSTEM
			}
		}
	}

	return true;
}

// [ref] run_inference_algorithm() in ${CPP_RND_HOME}/test/probabilistic_graphical_model/opengm/opengm_inference_algorithms.cpp
template<typename GraphicalModel, typename InferenceAlgorithm>
void run_inference_algorithm(InferenceAlgorithm &algorithm, std::vector<typename GraphicalModel::LabelType> &labelings)
{
	// optimize (approximately)
	typename InferenceAlgorithm::VerboseVisitorType visitor;
	//typename InferenceAlgorithm::TimingVisitorType visitor;
	//typename InferenceAlgorithm::EmptyVisitorType visitor;
	std::cout << "start inferring ..." << std::endl;
	{
		boost::timer::auto_cpu_timer timer;
		algorithm.infer(visitor);
	}
	std::cout << "end inferring ..." << std::endl;
	std::cout << "value: " << algorithm.value() << ", bound: " << algorithm.bound() << std::endl;

	// obtain the (approximate) argmax.
	algorithm.arg(labelings);
}

// [ref] normalize_histogram() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp
void normalize_histogram(cv::MatND &hist, const double factor)
{
#if 0
	// FIXME [modify] >>
	cvNormalizeHist(&(CvHistogram)hist, factor);
#else
	const cv::Scalar sums(cv::sum(hist));

	const double eps = 1.0e-20;
	if (std::fabs(sums[0]) < eps) return;

	//cv::Mat tmp(hist);
	//tmp.convertTo(hist, -1, factor / sums[0], 0.0);
	hist *= factor / sums[0];
#endif
}

}  // namespace local
}  // unnamed namespace

namespace my_interactive_graph_cuts {

}  // namespace my_interactive_graph_cuts

// [ref] "Interactive Graph Cuts for Optimal Boundary & Region Segmentation of Objects in N-D Images", Yuri Y. Boykov and Marie-Pierre Jolly, ICCV, 2001.
int interactive_graph_cuts_main(int argc, char *argv[])
{
	try
	{
#if 0
		const std::string input_image_filename("./segmentation_data/lattice_1.png");
		//const std::string input_image_filename("./segmentation_data/lattice_2.png");
		const std::string foreground_mask_filename("./segmentation_data/lattice_foreground.png");
		const std::string background_mask_filename("./segmentation_data/lattice_background_1.png");
		//const std::string background_mask_filename("./segmentation_data/lattice_background_2.png");
#elif 0
		const std::string input_image_filename("./segmentation_data/lattice_small_1.png");
		//const std::string input_image_filename("./segmentation_data/lattice_small_2.png");
		const std::string foreground_mask_filename("./segmentation_data/lattice_small_foreground.png");
		const std::string background_mask_filename("./segmentation_data/lattice_small_background_1.png");
		//const std::string background_mask_filename("./segmentation_data/lattice_small_background_2.png");
#elif 0
		const std::string input_image_filename("./segmentation_data/brain.png");
		//const std::string foreground_mask_filename("./segmentation_data/brain_foreground_1.png");
		const std::string foreground_mask_filename("./segmentation_data/brain_foreground_2.png");
		//const std::string background_mask_filename("./segmentation_data/brain_background_1.png");
		const std::string background_mask_filename("./segmentation_data/brain_background_2.png");
#elif 1
		const std::string input_image_filename("./segmentation_data/brain_small.png");
		//const std::string foreground_mask_filename("./segmentation_data/brain_small_foreground_1.png");
		const std::string foreground_mask_filename("./segmentation_data/brain_small_foreground_2.png");
		//const std::string background_mask_filename("./segmentation_data/brain_small_background_1.png");
		const std::string background_mask_filename("./segmentation_data/brain_small_background_2.png");
#endif

		const cv::Mat input_img(cv::imread(input_image_filename, CV_LOAD_IMAGE_GRAYSCALE));
		const cv::Mat foreground_mask(cv::imread(foreground_mask_filename, CV_LOAD_IMAGE_GRAYSCALE));
		const cv::Mat background_mask(cv::imread(background_mask_filename, CV_LOAD_IMAGE_GRAYSCALE));
		if (input_img.empty() || foreground_mask.empty() || background_mask.empty())
		{
			std::cout << "image file(s) not found" << std::endl;
			return 1;
		}

		// foreground & background probability distributions
		cv::MatND histForeground, histBackground;  // CV_32FC1, 1-dim (rows = bins, cols = 1)
		{
			const int dims = 1;
			const int bins = 256;
			const int histSize[] = { bins };
			const float range[] = { 0, 256 };
			const float *ranges[] = { range };
			const int channels[] = { 0 };

			// calculate histograms.
			cv::calcHist(
				&input_img, 1, channels, 0 == foreground_mask,
				histForeground, dims, histSize, ranges,
				true, // the histogram is uniform
				false
			);
			cv::calcHist(
				&input_img, 1, channels, 0 == background_mask,
				histBackground, dims, histSize, ranges,
				true, // the histogram is uniform
				false
			);

			// normalize histograms.
			local::normalize_histogram(histForeground, 1.0);
			local::normalize_histogram(histBackground, 1.0);
		}

		// create graphical model.
		local::GraphicalModel gm;
		const std::size_t numOfLabels = 2;
		const double lambda = 5.0;
		//const double sigma = 7.75264830;
		const double sigma = 1.0;
		if (local::create_graphical_model(input_img, numOfLabels, lambda, sigma, histForeground, histBackground, gm))
			std::cout << "A graphical model for interactive graph cuts is created." << std::endl;
		else
		{
			std::cout << "A graphical model for interactive graph cuts fails to be created." << std::endl;
			return 0;
		}

		// run inference algorithm.
		std::vector<local::GraphicalModel::LabelType> labelings(gm.numberOfVariables());
		{
#if 1
			typedef opengm::external::MinSTCutKolmogorov<std::size_t, double> MinStCut;
			typedef opengm::GraphCut<local::GraphicalModel, opengm::Minimizer, MinStCut> MinGraphCut;

			MinGraphCut mincut(gm);
#else
			typedef opengm::MinSTCutBoost<std::size_t, long, opengm::PUSH_RELABEL> MinStCut;
			typedef opengm::GraphCut<local::GraphicalModel, opengm::Minimizer, MinStCut> MinGraphCut;

			const MinGraphCut::ValueType scale = 1000000;
			const MinGraphCut::Parameter parameter(scale);
			MinGraphCut mincut(gm, parameter);
#endif

			local::run_inference_algorithm<local::GraphicalModel>(mincut, labelings);
		}

		// output results.
		{
#if 1
			cv::Mat label_img(input_img.size(), CV_8UC1, cv::Scalar::all(0));
			for (local::GraphicalModel::IndexType row = 0; row < (std::size_t)label_img.rows; ++row)
				for (local::GraphicalModel::IndexType col = 0; col < (std::size_t)label_img.cols; ++col)
					label_img.at<unsigned char>(row, col) = (unsigned char)(255 * labelings[local::getVariableIndex(label_img.cols, col, row)] / (numOfLabels - 1));

			cv::imshow("interactive graph cuts - labeling", label_img);
#elif 0
			cv::Mat label_img(input_img.size(), CV_16UC1, cv::Scalar::all(0));
			for (local::GraphicalModel::IndexType row = 0; row < label_img.rows; ++row)
				for (local::GraphicalModel::IndexType col = 0; col < label_img.cols; ++col)
					label_img.at<unsigned short>(row, col) = (unsigned short)labelings[local::getVariableIndex(label_img.cols, col, row)];

			cv::imshow("interactive graph cuts - labeling", label_img);
#else
			std::cout << algorithm.name() << " has found the labeling ";
			for (typename local::GraphicalModel::LabelType i = 0; i < labeling.size(); ++i)
				std::cout << labeling[i] << ' ';
			std::cout << std::endl;
#endif
		}

		cv::waitKey(0);

		cv::destroyAllWindows();
	}
	catch (const cv::Exception &e)
	{
		//std::cout << "OpenCV exception caught: " << e.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception caught:" << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;

		return 1;
	}

	return 0;
}
