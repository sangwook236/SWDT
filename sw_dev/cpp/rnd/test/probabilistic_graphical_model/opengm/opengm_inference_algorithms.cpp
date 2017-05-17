#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/graphicalmodel/space/grid_space.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/truncated_absolute_difference.hxx>
#include <opengm/functions/squared_difference.hxx>
#include <opengm/functions/truncated_squared_difference.hxx>
#include <opengm/functions/function_registration.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/dynamicprogramming.hxx>
#include <opengm/inference/astar.hxx>
#include <opengm/inference/dualdecomposition/dualdecomposition_subgradient.hxx>
#include <opengm/inference/dualdecomposition/dualdecomposition_bundle.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/icm.hxx>
#include <opengm/inference/loc.hxx>
#include <opengm/inference/gibbs.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/alphabetaswap.hxx>
#include <opengm/inference/icm.hxx>
#include <opengm/inference/lazyflipper.hxx>
#include <opengm/inference/loc.hxx>
#include <opengm/inference/gibbs.hxx>
#include <opengm/inference/swendsenwang.hxx>
#include <opengm/inference/bruteforce.hxx>
#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#include <opengm/inference/auxiliary/minstcutboost.hxx>
#include <opengm/inference/external/trws.hxx>
#include <opengm/inference/external/qpbo.hxx>
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>


#define __USE_GRID_SPACE 1
#define __USE_FULL_SIZE_IMAGE 1
#define __SAVE_RESULTANT_IMAGE 1

namespace {
namespace local {

#if defined(__USE_GRID_SPACE)
typedef opengm::GridSpace<std::size_t, std::size_t> Space;
#else
typedef opengm::SimpleDiscreteSpace<std::size_t, std::size_t> Space;
#endif
typedef opengm::ExplicitFunction<double> ExplicitFunction;
typedef opengm::PottsFunction<double> PottsFunction;
typedef opengm::TruncatedAbsoluteDifferenceFunction<double> TruncatedAbsoluteDifferenceFunction;
typedef opengm::SquaredDifferenceFunction<double> SquaredDifferenceFunction;
typedef opengm::TruncatedSquaredDifferenceFunction<double> TruncatedSquaredDifferenceFunction;

// construct a graphical model with
// - addition as the operation (template parameter Adder)
// - support for Potts functions (template parameter PottsFunction<double>)
typedef opengm::GraphicalModel<double, opengm::Adder, OPENGM_TYPELIST_2(ExplicitFunction, PottsFunction), Space> GraphicalModelForPottsModel;

#if defined(__SAVE_RESULTANT_IMAGE)
std::size_t NUM_COLS = 0;
std::size_t NUM_ROWS = 0;
std::size_t NUM_LABELS = 0;
#endif

// this function maps a node (x, y) in the grid to a unique variable index
inline std::size_t getVariableIndex(const std::size_t Nx, const std::size_t x, const std::size_t y)
{
	return x + Nx * y;
}

// REF [file] >> ${OPENGM_HOME}/src/examples/image-processing-examples/grid_potts.cxx.
// REF [paper] >> "Efficient Belief Propagation for Early Vision", P. F. Felzenszwalb & D. Huttenlocher, IJCV, 2006.
bool createGraphicalModelForPottsModel(GraphicalModelForPottsModel &gm)
{
	const std::string img_filename("./data/probabilistic_graphical_model/teddy-imL.png");

	const cv::Mat &img = cv::imread(img_filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (img.empty())
	{
		std::cout << "image file not found" << std::endl;
		return false;
	}

	// model parameters (global variables are used only in example code)
#if defined(__USE_FULL_SIZE_IMAGE)
	const std::size_t Nx = img.cols;  // width of the grid
	const std::size_t Ny = img.rows;  // height of the grid
#else
	const std::size_t Nx = 30;  // width of the grid
	const std::size_t Ny = 30;  // height of the grid
#endif
	const std::size_t numOfLabels = 5;
	const double lambda = 0.1;  // coupling strength of the Potts model

#if defined(__SAVE_RESULTANT_IMAGE)
	NUM_COLS = Nx;
	NUM_ROWS = Ny;
	NUM_LABELS = numOfLabels;
#endif

	// construct a label space with
	// - Nx * Ny variables
	// - each having numOfLabels many labels
#if defined(__USE_GRID_SPACE)
	Space space(Nx, Ny, numOfLabels);
#else
	Space space(Nx * Ny, numOfLabels);
#endif

	gm = GraphicalModelForPottsModel(space);

	{
		// for each node (x, y) in the grid, i.e. for each variable variableIndex(Nx, x, y) of the model,
		// add one 1st order functions and one 1st order factor
		for (std::size_t y = 0; y < Ny; ++y)
			for (std::size_t x = 0; x < Nx; ++x)
			{
				// function
				const std::size_t shape[] = { numOfLabels };
				ExplicitFunction func1(shape, shape + 1);
				for (std::size_t state = 0; state < numOfLabels; ++state)
					// FIXME [check] >> is it correct?
					func1(state) = (1.0 - lambda) * state / (numOfLabels - 1);

				const GraphicalModelForPottsModel::FunctionIdentifier fid1 = gm.addFunction(func1);

				// factor
				const std::size_t variableIndices[] = { getVariableIndex(Nx, x, y) };
				gm.addFactor(fid1, variableIndices, variableIndices + 1);
			}
	}

	{
		// add one (!) 2nd order Potts function
		PottsFunction func2(numOfLabels, numOfLabels, 0.0, lambda);
		const GraphicalModelForPottsModel::FunctionIdentifier fid2 = gm.addFunction(func2);

		// for each pair of nodes (x1, y1), (x2, y2) which are adjacent on the grid,
		// add one factor that connects the corresponding variable indices and refers to the Potts function
		for (std::size_t y = 0; y < Ny; ++y)
			for (std::size_t x = 0; x < Nx; ++x)
			{
				if (x + 1 < Nx)  // (x, y) -- (x + 1, y)
				{
					std::size_t variableIndices[] = { getVariableIndex(Nx, x, y), getVariableIndex(Nx, x + 1, y) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				if (y + 1 < Ny)  // (x, y) -- (x, y + 1)
				{
					std::size_t variableIndices[] = { getVariableIndex(Nx, x, y), getVariableIndex(Nx, x, y + 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
			}
	}

	return true;
}

#if 1

typedef opengm::GraphicalModel<double, opengm::Adder, ExplicitFunction, Space> GraphicalModelForStereoMatching;
typedef opengm::GraphicalModel<double, opengm::Adder, ExplicitFunction, Space> GraphicalModelForImageRestoration;

// REF [paper] >> "Efficient Belief Propagation for Early Vision", P. F. Felzenszwalb & D. Huttenlocher, IJCV, 2006.
bool createGraphicalModelForStereoMatching(GraphicalModelForStereoMatching &gm)
{
	const std::string imgL_filename("./data/probabilistic_graphical_model/tsukuba-imL.png");
	const std::string imgR_filename("./data/probabilistic_graphical_model/tsukuba-imR.png");
	const std::string imgT_filename("./data/probabilistic_graphical_model/tsukuba-truedispLR.png");

	const cv::Mat &imgL = cv::imread(imgL_filename, CV_LOAD_IMAGE_GRAYSCALE);
	const cv::Mat &imgR = cv::imread(imgR_filename, CV_LOAD_IMAGE_GRAYSCALE);
	//const cv::Mat &imgT = cv::imread(imgT_filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (imgL.empty() || imgR.empty())
	{
		std::cout << "image files not found" << std::endl;
		return false;
	}

	// REF [paper] >> "Efficient Belief Propagation for Early Vision", P. F. Felzenszwalb & D. Huttenlocher, IJCV, 2006.
	const std::size_t disparities = 20;
	const double tau = 15.0;
	const double lambda = 0.07;
	const double d = 1.7;

	// model parameters (global variables are used only in example code)
#if defined(__USE_FULL_SIZE_IMAGE)
	const std::size_t Nx = imgL.cols;  // width of the grid
	const std::size_t Ny = imgL.rows;  // height of the grid
#else
	const std::size_t Nx = 30;  // width of the grid
	const std::size_t Ny = 30;  // height of the grid
#endif
	const std::size_t numOfLabels = disparities;

#if defined(__SAVE_RESULTANT_IMAGE)
	NUM_COLS = Nx;
	NUM_ROWS = Ny;
	NUM_LABELS = numOfLabels;
#endif

	// construct a label space with
	// - Nx * Ny variables
	// - each having numOfLabels many labels
#if defined(__USE_GRID_SPACE)
	Space space(Nx, Ny, numOfLabels);
#else
	Space space(Nx * Ny, numOfLabels);
#endif

	// construct a graphical model with
	// - addition as the operation (template parameter Adder)
	// - support for truncated absolute difference functions (template parameter TruncatedAbsoluteDifferenceFunction<double>)
	gm = GraphicalModelForStereoMatching(space);

	{
		const double threshold = tau;
		const double weight = lambda;

		// add 1st order functions and factors to the model
		for (std::size_t y = 0; y < Ny; ++y)
		{
			for (std::size_t x = 0; x < Nx; ++x)
			{
				const std::size_t var = getVariableIndex(Nx, x, y);

				// construct 1st order function
				//const std::size_t shape[] = { gm.numberOfLabels(var) };
				const std::size_t shape[] = { numOfLabels };
				ExplicitFunction func1(shape, shape + 1);
				for (std::size_t state = 0; state < numOfLabels; ++state)
				{
#if 1
					if ((long)x - (long)state >= 0)
						func1(state) = weight * std::min(std::abs((double)imgL.at<unsigned char>(y, x) - (double)imgR.at<unsigned char>(y, x - state)), threshold);
#else
					if (x + state < imgR.cols)
						func1(state) = weight * std::min(std::abs((double)imgL.at<unsigned char>(y, x) - (double)imgR.at<unsigned char>(y, x + state)), threshold);
#endif
					else func1(state) = std::numeric_limits<double>::max();
				}

				// add function
				const GraphicalModelForStereoMatching::FunctionIdentifier fid1 = gm.addFunction(func1);

				// add factor
				const std::size_t variableIndices[] = { var };
				gm.addFactor(fid1, variableIndices, variableIndices + 1);
			}
		}
	}

	{
		const double threshold = d;
		const double weight = 1.0;

		// construct 2nd order function
		const std::size_t shape[] = { numOfLabels, numOfLabels };
		ExplicitFunction func2(shape, shape + 2);
		for (std::size_t state1 = 0; state1 < numOfLabels; ++state1)
			for (std::size_t state2 = 0; state2 < numOfLabels; ++state2)
			{
				const double diff = (double)state1 - (double)state2;
				func2(state1, state2) = weight * std::min(std::abs(diff), threshold);
			}

		// add 2nd order functions and factors to the model
		const GraphicalModelForStereoMatching::FunctionIdentifier fid2 = gm.addFunction(func2);
		for (std::size_t y = 0; y < Ny; ++y)
			for (std::size_t x = 0; x < Nx; ++x)
			{
				// add factor
				const std::size_t var = getVariableIndex(Nx, x, y);
				/*
				if ((long)x - 1 >= 0)  // (x - 1, y) -- (x, y)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x - 1, y) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				if ((long)y - 1 >= 0)  // (x, y - 1) -- (x, y)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x, y - 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				*/
				if (x + 1 < Nx)  // (x, y) -- (x + 1, y)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x + 1, y) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				if (y + 1 < Ny)  // (x, y) -- (x, y + 1)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x, y + 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
			}
	}

	return true;
}

// REF [paper] >> "Efficient Belief Propagation for Early Vision", P. F. Felzenszwalb & D. Huttenlocher, IJCV, 2006.
bool createGraphicalModelForImageRestoration(GraphicalModelForImageRestoration &gm)
{
	const std::string img_filename("./data/probabilistic_graphical_model/penguin-input.png");
	const std::string imgT_filename("./data/probabilistic_graphical_model/penguin-mask.png");

	const cv::Mat &img = cv::imread(img_filename, CV_LOAD_IMAGE_GRAYSCALE);
	//const cv::Mat &imgT = cv::imread(imgT_filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (img.empty())
	{
		std::cout << "image file not found" << std::endl;
		return false;
	}

	// REF [paper] >> "Efficient Belief Propagation for Early Vision", P. F. Felzenszwalb & D. Huttenlocher, IJCV, 2006.
	const std::size_t intensities = 256;  // [0 255]
	const double lambda = 0.04;
	const double d = 200;

	// model parameters (global variables are used only in example code)
#if defined(__USE_FULL_SIZE_IMAGE)
	const std::size_t Nx = img.cols;  // width of the grid
	const std::size_t Ny = img.rows;  // height of the grid
#else
	const std::size_t Nx = 30;  // width of the grid
	const std::size_t Ny = 30;  // height of the grid
#endif
	const std::size_t numOfLabels = intensities;

#if defined(__SAVE_RESULTANT_IMAGE)
	NUM_COLS = Nx;
	NUM_ROWS = Ny;
	NUM_LABELS = numOfLabels;
#endif

	// construct a label space with
	// - Nx * Ny variables
	// - each having numOfLabels many labels
#if defined(__USE_GRID_SPACE)
	Space space(Nx, Ny, numOfLabels);
#else
	Space space(Nx * Ny, numOfLabels);
#endif

	// construct a graphical model with
	// - addition as the operation (template parameter Adder)
	// - support for sqaured difference & truncated sqaured difference functions (template parameter SquaredDifferenceFunction<double> & TruncatedSqauredDifferenceFunction)
	gm = GraphicalModelForImageRestoration(space);

	{
		const double weight = lambda;

		// add 1st order functions and factors to the model
		for (std::size_t y = 0; y < Ny; ++y)
		{
			for (std::size_t x = 0; x < Nx; ++x)
			{
				const std::size_t var = getVariableIndex(Nx, x, y);

				// construct 1st order function
				//const std::size_t shape[] = { gm.numberOfLabels(var) };
				const std::size_t shape[] = { numOfLabels };
				ExplicitFunction func1(shape, shape + 1);
				for (std::size_t state = 0; state < numOfLabels; ++state)
				{
					const double diff = img.at<unsigned char>(y, x) - state;
					func1(state) = weight * diff * diff;
				}

				// add function
				const GraphicalModelForStereoMatching::FunctionIdentifier fid1 = gm.addFunction(func1);

				// add factor
				const std::size_t variableIndices[] = { var };
				gm.addFactor(fid1, variableIndices, variableIndices + 1);
			}
		}
	}

	{
		const double threshold = d;
		const double weight = 1.0;

		// construct 2nd order function
		const std::size_t shape[] = { numOfLabels, numOfLabels };
		ExplicitFunction func2(shape, shape + 2);
		for (std::size_t state1 = 0; state1 < numOfLabels; ++state1)
			for (std::size_t state2 = 0; state2 < numOfLabels; ++state2)
			{
				const double diff = (double)state1 - (double)state2;
				func2(state1, state2) = weight * std::min<double>(diff * diff, threshold);
			}

		// add 2nd order functions and factors to the model
		const GraphicalModelForStereoMatching::FunctionIdentifier fid2 = gm.addFunction(func2);
		for (std::size_t y = 0; y < Ny; ++y)
			for (std::size_t x = 0; x < Nx; ++x)
			{
				// add factor
				const std::size_t var = getVariableIndex(Nx, x, y);
				/*
				if ((long)x - 1 >= 0)  // (x - 1, y) -- (x, y)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x - 1, y) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				if ((long)y - 1 >= 0)  // (x, y - 1) -- (x, y)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x, y - 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				*/
				if (x + 1 < Nx)  // (x, y) -- (x + 1, y)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x + 1, y) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				if (y + 1 < Ny)  // (x, y) -- (x, y + 1)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x, y + 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
			}
	}

	return true;
}

#else

// REF [file] >> ${OPENGM_HOME}/src/examples/image-processing-examples/interpixel_boundary_segmentation.cxx.
struct TruncatedAbsoluteDifferenceFunctionForStereoMatching : opengm::FunctionBase<TruncatedAbsoluteDifferenceFunctionForStereoMatching, double, std::size_t, std::size_t>
{
public:
	TruncatedAbsoluteDifferenceFunctionForStereoMatching(cv::Mat &imgL, cv::Mat &imgR, const std::size_t x, const std::size_t y, const std::size_t numOfLabels, const double threshold, const double weight)
	: imgL_(&imgL), imgR_(&imgR), x_(x), y_(y), numOfLabels_(numOfLabels), threshold_(threshold), weight_(weight)
	{
		assert(imgL_ && imgL_->rows > 0 && imgL_->cols > 0);
		assert(imgR_ && imgR_->rows > 0 && imgR_->cols > 0);
		assert(imgL_->rows == imgR_->rows && imgL_->cols == imgR_->cols);
	}

	explicit TruncatedAbsoluteDifferenceFunctionForStereoMatching(const TruncatedAbsoluteDifferenceFunctionForStereoMatching &rhs)
	: imgL_(rhs.imgL_), imgR_(rhs.imgR_), x_(rhs.x_), y_(rhs.y_), numOfLabels_(rhs.numOfLabels_), threshold_(rhs.threshold_), weight_(rhs.weight_)
	{
		assert(imgL_ && imgL_->rows > 0 && imgL_->cols > 0);
		assert(imgR_ && imgR_->rows > 0 && imgR_->cols > 0);
		assert(imgL_->rows == imgR_->rows && imgL_->cols == imgR_->cols);
	}

	TruncatedAbsoluteDifferenceFunctionForStereoMatching & operator=(const TruncatedAbsoluteDifferenceFunctionForStereoMatching &rhs)
	{
		if (&rhs == this) return *this;

		imgL_ = rhs.imgL_;
		imgR_ = rhs.imgR_;
		x_ = rhs.x_;
		y_ = rhs.y_;
		numOfLabels_ = rhs.numOfLabels_;
		threshold_ = rhs.threshold_;
		weight_ = rhs.weight_;

		return *this;
	}

	template<class Iterator>
	inline const double operator()(Iterator begin) const
	{
		const std::size_t &state = *begin;

		// FIXME [check] >> which one is correct?
#if 1
		if ((long)x_ - (long)state >= 0)
			return weight_ * std::min(std::abs((double)imgL_->at<unsigned char>(y_, x_) - (double)imgR_->at<unsigned char>(y_, x_ - state)), threshold_);
#else
		if (x_ + state < imgR_->cols)
			return weight_ * std::min(std::abs((double)imgL_->at<unsigned char>(y_, x_) - (double)imgR_->at<unsigned char>(y_, x_ + state)), threshold_);
#endif
		else return std::numeric_limits<double>::max();
	}

	inline std::size_t dimension() const
	{
		return 1;
	}

	inline std::size_t shape(const std::size_t idx) const
	{
		return numOfLabels_;
	}

	inline std::size_t size() const
	{
		return numOfLabels_;
	}

private:
	cv::Mat *imgL_, *imgR_;

	std::size_t x_, y_;
	std::size_t numOfLabels_;
	double threshold_, weight_;
};

// REF [file] >> ${OPENGM_HOME}/src/examples/image-processing-examples/interpixel_boundary_segmentation.cxx.
struct SquaredDifferenceFunctionForImageRestoration : opengm::FunctionBase<SquaredDifferenceFunctionForImageRestoration, double, std::size_t, std::size_t>
{
public:
	SquaredDifferenceFunctionForImageRestoration(cv::Mat &img, const std::size_t x, const std::size_t y, const std::size_t numOfLabels, const double weight)
	: img_(&img), x_(x), y_(y), numOfLabels_(numOfLabels), weight_(weight)
	{
		assert(img_ && img_->rows > 0 && img_->cols > 0);
	}

	explicit SquaredDifferenceFunctionForImageRestoration(const SquaredDifferenceFunctionForImageRestoration &rhs)
	: img_(rhs.img_), x_(rhs.x_), y_(rhs.y_), numOfLabels_(rhs.numOfLabels_), weight_(rhs.weight_)
	{
		assert(img_ && img_->rows > 0 && img_->cols > 0);
	}

	SquaredDifferenceFunctionForImageRestoration & operator=(const SquaredDifferenceFunctionForImageRestoration &rhs)
	{
		if (&rhs == this) return *this;

		img_ = rhs.img_;
		x_ = rhs.x_;
		y_ = rhs.y_;
		numOfLabels_ = rhs.numOfLabels_;
		weight_ = rhs.weight_;

		return *this;
	}

	template<class Iterator>
	inline const double operator()(Iterator begin) const
	{
		const std::size_t &state = *begin;

		const double diff = (double)img_->at<unsigned char>(y_, x_) - (double)state;
		return weight_ * diff * diff;
	}

	std::size_t dimension() const
	{
		return 1;
	}

	std::size_t shape(const std::size_t idx) const
	{
		return numOfLabels_;
	}

	std::size_t size() const
	{
		return numOfLabels_;
	}

private:
	cv::Mat *img_;

	std::size_t x_, y_;
	std::size_t numOfLabels_;
	double weight_;
};

//typedef OPENGM_TYPELIST_2(TruncatedAbsoluteDifferenceFunctionForStereoMatching, TruncatedAbsoluteDifferenceFunction) FunctionTypelistForStereoMatching;
typedef opengm::meta::TypeListGenerator<TruncatedAbsoluteDifferenceFunctionForStereoMatching, TruncatedAbsoluteDifferenceFunction>::type FunctionTypelistForStereoMatching;
typedef opengm::GraphicalModel<double, opengm::Adder, FunctionTypelistForStereoMatching, Space> GraphicalModelForStereoMatching;

//typedef OPENGM_TYPELIST_2(SquaredDifferenceFunctionForImageRestoration, TruncatedSquaredDifferenceFunction) FunctionTypelistForImageRestoration;
typedef opengm::meta::TypeListGenerator<SquaredDifferenceFunctionForImageRestoration, TruncatedSquaredDifferenceFunction>::type FunctionTypelistForImageRestoration;
typedef opengm::GraphicalModel<double, opengm::Adder, FunctionTypelistForImageRestoration, Space> GraphicalModelForImageRestoration;

// REF [paper] >> "Efficient Belief Propagation for Early Vision", P. F. Felzenszwalb & D. Huttenlocher, IJCV, 2006.
bool createGraphicalModelForStereoMatching(GraphicalModelForStereoMatching &gm)
{
	const std::string imgL_filename("./data/probabilistic_graphical_model/tsukuba-imL.png");
	const std::string imgR_filename("./data/probabilistic_graphical_model/tsukuba-imR.png");
	const std::string imgT_filename("./data/probabilistic_graphical_model/tsukuba-truedispLR.png");

	cv::Mat &imgL = cv::imread(imgL_filename, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat &imgR = cv::imread(imgR_filename, CV_LOAD_IMAGE_GRAYSCALE);
	//const cv::Mat &imgT = cv::imread(imgT_filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (imgL.empty() || imgR.empty())
	{
		std::cout << "image files not found" << std::endl;
		return false;
	}

	// REF [paper] >> "Efficient Belief Propagation for Early Vision", P. F. Felzenszwalb & D. Huttenlocher, IJCV, 2006.
	const std::size_t disparities = 20;
	const double tau = 15.0;
	const double lambda = 0.07;
	const double d = 1.7;

	// model parameters (global variables are used only in example code)
#if defined(__USE_FULL_SIZE_IMAGE)
	const std::size_t Nx = imgL.cols;  // width of the grid
	const std::size_t Ny = imgL.rows;  // height of the grid
#else
	const std::size_t Nx = 30;  // width of the grid
	const std::size_t Ny = 30;  // height of the grid
#endif
	const std::size_t numOfLabels = disparities;

#if defined(__SAVE_RESULTANT_IMAGE)
	NUM_COLS = Nx;
	NUM_ROWS = Ny;
	NUM_LABELS = numOfLabels;
#endif

	// construct a label space with
	// - Nx * Ny variables
	// - each having numOfLabels many labels
#if defined(__USE_GRID_SPACE)
	Space space(Nx, Ny, numOfLabels);
#else
	Space space(Nx * Ny, numOfLabels);
#endif

	// construct a graphical model with
	// - addition as the operation (template parameter Adder)
	// - support for truncated absolute difference functions (template parameter TruncatedAbsoluteDifferenceFunction<double>)
	gm = GraphicalModelForStereoMatching(space);

	{
		// for each node (x, y) in the grid, i.e. for each variable variableIndex(Nx, x, y) of the model,
		// add one 1st order functions and one 1st order factor
		const double threshold = tau;
		const double weight = lambda;
		for (std::size_t y = 0; y < Ny; ++y)
		{
			for (std::size_t x = 0; x < Nx; ++x)
			{
				TruncatedAbsoluteDifferenceFunctionForStereoMatching func1(imgL, imgR, x, y, numOfLabels, threshold, weight);
				const GraphicalModelForStereoMatching::FunctionIdentifier fid1 = gm.addFunction(func1);

				const std::size_t variableIndices[] = { getVariableIndex(Nx, x, y) };
				gm.addFactor(fid1, variableIndices, variableIndices + 1);
			}
		}
	}

	{
		// add one 2nd order functions and one 2nd order factor
		const double threshold = d;
		const double weight = 1.0;
		TruncatedAbsoluteDifferenceFunction func2(numOfLabels, numOfLabels, threshold, weight);
		const GraphicalModelForStereoMatching::FunctionIdentifier fid2 = gm.addFunction(func2);

		// for each pair of nodes (x1, y1), (x2, y2) which are adjacent on the grid,
		// add one factor that connects the corresponding variable indices
		for (std::size_t y = 0; y < Ny; ++y)
			for (std::size_t x = 0; x < Nx; ++x)
			{
				const std::size_t var = getVariableIndex(Nx, x, y);
				/*
				if ((long)x - 1 >= 0)  // (x - 1, y) -- (x, y)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x - 1, y) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				if ((long)y - 1 >= 0)  // (x, y - 1) -- (x, y)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x, y - 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				*/
				if (x + 1 < Nx)  // (x, y) -- (x + 1, y)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x + 1, y) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				if (y + 1 < Ny)  // (x, y) -- (x, y + 1)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x, y + 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
			}
	}

	return true;
}

// REF [paper] >> "Efficient Belief Propagation for Early Vision", P. F. Felzenszwalb & D. Huttenlocher, IJCV, 2006.
bool createGraphicalModelForImageRestoration(GraphicalModelForImageRestoration &gm)
{
	const std::string img_filename("./data/probabilistic_graphical_model/penguin-input.png.png");
	const std::string imgT_filename("./data/probabilistic_graphical_model/penguin-mask.png");

	cv::Mat &img = cv::imread(img_filename, CV_LOAD_IMAGE_GRAYSCALE);
	//const cv::Mat &imgT = cv::imread(imgT_filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (img.empty())
	{
		std::cout << "image file not found" << std::endl;
		return false;
	}

	// REF [paper] >> "Efficient Belief Propagation for Early Vision", P. F. Felzenszwalb & D. Huttenlocher, IJCV, 2006.
	const std::size_t intensities = 256;  // [0 255]
	const double lambda = 0.04;
	const double d = 200;

	// model parameters (global variables are used only in example code)
#if defined(__USE_FULL_SIZE_IMAGE)
	const std::size_t Nx = imgL.cols;  // width of the grid
	const std::size_t Ny = imgL.rows;  // height of the grid
#else
	const std::size_t Nx = 30;  // width of the grid
	const std::size_t Ny = 30;  // height of the grid
#endif
	const std::size_t numOfLabels = intensities;

#if defined(__SAVE_RESULTANT_IMAGE)
	NUM_COLS = Nx;
	NUM_ROWS = Ny;
	NUM_LABELS = numOfLabels;
#endif

	// construct a label space with
	// - Nx * Ny variables
	// - each having numOfLabels many labels
#if defined(__USE_GRID_SPACE)
	Space space(Nx, Ny, numOfLabels);
#else
	Space space(Nx * Ny, numOfLabels);
#endif

	// construct a graphical model with
	// - addition as the operation (template parameter Adder)
	// - support for sqaured difference & truncated sqaured difference functions (template parameter SquaredDifferenceFunction<double> & TruncatedSqauredDifferenceFunction)
	gm = GraphicalModelForImageRestoration(space);

	{
		// for each node (x, y) in the grid, i.e. for each variable variableIndex(Nx, x, y) of the model,
		// add one 1st order functions and one 1st order factor
		const double weight = lambda;
		for (std::size_t y = 0; y < Ny; ++y)
			for (std::size_t x = 0; x < Nx; ++x)
			{
				const SquaredDifferenceFunctionForImageRestoration func1(img, x, y, numOfLabels, weight);
				const GraphicalModelForImageRestoration::FunctionIdentifier fid1 = gm.addFunction(func1);

				const std::size_t variableIndices[] = { getVariableIndex(Nx, x, y) };
				gm.addFactor(fid1, variableIndices, variableIndices + 1);
			}
	}

	{
		// add one 1st order functions and one 1st order factor
		const double threshold = d;
		const double weight = 1.0;
		const TruncatedSquaredDifferenceFunction func2(numOfLabels, numOfLabels, threshold, weight);
		const GraphicalModelForImageRestoration::FunctionIdentifier fid2 = gm.addFunction(func2);

		// for each pair of nodes (x1, y1), (x2, y2) which are adjacent on the grid,
		// add one factor that connects the corresponding variable indices
		for (std::size_t y = 0; y < Ny; ++y)
			for (std::size_t x = 0; x < Nx; ++x)
			{
				const std::size_t var = getVariableIndex(Nx, x, y);
				/*
				if ((long)x - 1 >= 0)  // (x - 1, y) -- (x, y)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x - 1, y) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				if ((long)y - 1 >= 0)  // (x, y - 1) -- (x, y)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x, y - 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				*/
				if (x + 1 < Nx)  // (x, y) -- (x + 1, y)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x + 1, y) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				if (y + 1 < Ny)  // (x, y) -- (x, y + 1)
				{
					std::size_t variableIndices[] = { var, getVariableIndex(Nx, x, y + 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
			}
	}

	return true;
}

#endif

template<typename GraphicalModel, typename InferenceAlgorithm>
void run_inference_algorithm(const std::size_t numOfVariables, InferenceAlgorithm &algorithm, const std::string &output_filename)
{
	// Optimize (approximately).
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

	// Obtain the (approximate) argmin/argmax.
	std::vector<typename GraphicalModel::LabelType> labeling(numOfVariables);
	algorithm.arg(labeling);

	//
#if defined(__SAVE_RESULTANT_IMAGE)
	if (!output_filename.empty())
	{
		//NUM_COLS = Nx;
		//NUM_ROWS = Ny;
		//NUM_LABELS = numOfLabels;

		cv::Mat img(NUM_ROWS, NUM_COLS, CV_8UC1, cv::Scalar::all(0));

		for (typename GraphicalModel::IndexType row = 0; row < NUM_ROWS; ++row)
			for (typename GraphicalModel::IndexType col = 0; col < NUM_COLS; ++col)
				img.at<unsigned char>(row, col) = (unsigned char)(255 * labeling[getVariableIndex(NUM_COLS, col, row)] / (NUM_LABELS - 1));

		cv::imwrite("./data/probabilistic_graphical_model/opengm/" + output_filename + "_result.png", img);
	}
	else
		std::cout << "error: file name is empty" << std::endl;
#else
	std::cout << algorithm.name() << " has found the labeling ";
	for (typename GraphicalModel::LabelType i = 0; i < labeling.size(); ++i)
		std::cout << labeling[i] << ' ';
	std::cout << std::endl;
#endif
}

template<typename GraphicalModel>
void inference_algorithms(GraphicalModel &gm, const std::string &problem_name)
{
	const bool isBinaryVariable = false;
	const std::size_t MAX_ITERATIONS = 50;
	const double CONVERGENCE_BOUND = 1e-7;

	const bool runBP = true;
	const bool runTRBP = true;
	const bool runTRWS = true;
	const bool runLP = false;  // Not yet implemented.
	const bool runDP = false;
	const bool runDualDecomposition = true;
	const bool runAStar = false;
	const bool runGraphCuts = isBinaryVariable;  // For binary variables only.
	const bool runQPBO = isBinaryVariable;  // For binary variables only.
	const bool runExpansionMove = true;
	const bool runSwapMove = true;
	const bool runICM = true;
	const bool runLazyFlipper = true;
	const bool runLOC = true;
	const bool runnGibbs = false;  // Not yet implemented.
	const bool runSwendsenWang = false;  // Not yet implemented.
	const bool runBruteforce = false;

	// Inference algorithms.
	{
		std::cout << "\nBelief propagation (BP) algorithm -----------------------------------" << std::endl;
		if (runBP)
		{
			const std::string inf_name("_bp");

			// Set up the optimizer (loopy belief propagation).
			typedef opengm::BeliefPropagationUpdateRules<GraphicalModel, opengm::Minimizer> UpdateRules;
			typedef opengm::MessagePassing<GraphicalModel, opengm::Minimizer, UpdateRules, opengm::MaxDistance> BeliefPropagation;

			const std::size_t maxNumberOfIterations = MAX_ITERATIONS;
			const double convergenceBound = CONVERGENCE_BOUND;
			const double damping = 0.0;
			const typename BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
			BeliefPropagation bp(gm, parameter);

			run_inference_algorithm<GraphicalModel>(gm.numberOfVariables(), bp, problem_name + inf_name);
		}

		std::cout << "\ntree-reweighted belief propagation (TRBP) algorithm -----------------" << std::endl;
		if (runTRBP)
		{
			const std::string inf_name("_trbp");

			// Set up the optimizer (tree re-weighted belief propagation).
			typedef opengm::TrbpUpdateRules<GraphicalModel, opengm::Minimizer> UpdateRules;
			typedef opengm::MessagePassing<GraphicalModel, opengm::Minimizer, UpdateRules, opengm::MaxDistance> TRBP;

			const std::size_t maxNumberOfIterations = MAX_ITERATIONS;
			const double convergenceBound = CONVERGENCE_BOUND;
			const double damping = 0.0;
			const typename TRBP::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
			TRBP trbp(gm, parameter);

			run_inference_algorithm<GraphicalModel>(gm.numberOfVariables(), trbp, problem_name + inf_name);
		}

		std::cout << "\nsequential tree-reweighted message passing (TRW-S) algorithm --------" << std::endl;
		if (runTRWS)
		{
			const std::string inf_name("_trws");

			// Set up the optimizer (tree re-weighted belief propagation).
			typedef opengm::external::TRWS<GraphicalModel> TRWS;

			typename TRWS::Parameter parameter;
			parameter.numberOfIterations_ = MAX_ITERATIONS;
			parameter.useRandomStart_ = false;
			parameter.doBPS_ = false;
			parameter.energyType_ = TRWS::Parameter::TL1;
			parameter.tolerance_ = CONVERGENCE_BOUND;
			TRWS trws(gm, parameter);

			run_inference_algorithm<GraphicalModel>(gm.numberOfVariables(), trws, problem_name + inf_name);
		}

		std::cout << "\n(Integer) linear programming (ILP) ----------------------------------" << std::endl;
		if (runLP)
		{
			throw std::runtime_error("Not yet implemented");
		}

		std::cout << "\nDynamic programming (DP) --------------------------------------------" << std::endl;
		if (runDP)
		{
			const std::string inf_name("_dp");

			// Set up the optimizer.
			typedef opengm::DynamicProgramming<GraphicalModel, opengm::Minimizer> DP;

			DP dp(gm);

			run_inference_algorithm<GraphicalModel>(gm.numberOfVariables(), dp, problem_name + inf_name);
		}

		std::cout << "\nDual decomposition algorithm ----------------------------------------" << std::endl;
		if (runDualDecomposition)
		{
			const std::string inf_name("_dd");

			// Set up the optimizer.
			typedef opengm::DDDualVariableBlock<marray::Marray<typename GraphicalModel::ValueType> > DualBlock;
			typedef typename opengm::DualDecompositionBase<GraphicalModel, DualBlock>::SubGmType SubGraphicalModel;
			typedef opengm::BeliefPropagationUpdateRules<SubGraphicalModel, opengm::Minimizer> UpdateRule;
			typedef opengm::MessagePassing<SubGraphicalModel, opengm::Minimizer, UpdateRule, opengm::MaxDistance> InferenceEngine;
#if 1
			typedef opengm::DualDecompositionSubGradient<GraphicalModel, InferenceEngine, DualBlock> DualDecomposition;
#else
			// not yet supported
			typedef opengm::DualDecompositionBundle<GraphicalModel, InferenceEngine, DualBlock> DualDecomposition;
#endif

			const std::size_t maxNumberOfIterations = MAX_ITERATIONS;
			const double convergenceBound = CONVERGENCE_BOUND;
			const double damping = 0.0;
			const typename InferenceEngine::Parameter InfParameter(maxNumberOfIterations, convergenceBound, damping);
			typename DualDecomposition::Parameter parameter;
			parameter.subPara_ = InfParameter;
			parameter.useAdaptiveStepsize_ = false;
			parameter.useProjectedAdaptiveStepsize_ = false;
			DualDecomposition dd(gm, parameter);

			run_inference_algorithm<GraphicalModel>(gm.numberOfVariables(), dd, problem_name + inf_name);
		}

		std::cout << "\nA-star search algorithm ---------------------------------------------" << std::endl;
		if (runAStar)
		{
			const std::string inf_name("_astar");

			// Set up the optimizer.
			typedef opengm::AStar<GraphicalModel, opengm::Minimizer> AStar;

			typename AStar::Parameter parameter;
			parameter.maxHeapSize_ = 3000000;
			parameter.numberOfOpt_ = 1;
			parameter.objectiveBound_ = CONVERGENCE_BOUND;
			parameter.heuristic_ = AStar::Parameter::FASTHEURISTIC;
			//parameter.nodeOrder_ = ...;
			//parameter.treeFactorIds_ = ...;
			AStar astar(gm, parameter);

			run_inference_algorithm<GraphicalModel>(gm.numberOfVariables(), astar, problem_name + inf_name);
		}

		std::cout << "\nGraph-cuts algorithm ------------------------------------------------" << std::endl;
		if (runGraphCuts)
		{
			// NOTIC [caution] >> this implementation of the graph-cuts supports only binary variables.

			const std::string inf_name("_graphcuts");

			// Set up the optimizer.
#if 1
			typedef opengm::external::MinSTCutKolmogorov<std::size_t, double> MinStCut;
#else
			typedef opengm::MinSTCutBoost<std::size_t, long, opengm::PUSH_RELABEL> MinStCut;
#endif
			typedef opengm::GraphCut<GraphicalModel, opengm::Minimizer, MinStCut> MinGraphCut;

			const typename MinGraphCut::ValueType scale = 1000000;
			const typename MinGraphCut::Parameter parameter(scale);
			MinGraphCut mincut(gm, parameter);

			run_inference_algorithm<GraphicalModel>(gm.numberOfVariables(), mincut, problem_name + inf_name);
		}

		std::cout << "\nquadratic pseudo-boolean optimization (QPBO) algorithm --------------" << std::endl;
		if (runQPBO)
		{
			// NOTICE [caution] >> this implementation of QPBO supports only binary variables.

			const std::string inf_name("_qpbo");

			// Set up the optimizer.
			typedef opengm::external::QPBO<GraphicalModel> QPBO;

			typename QPBO::Parameter parameter;
			parameter.strongPersistency_ = true;  // Forcing strong persistency.
			parameter.useImproveing_ = false;  // Using improving technique.
			parameter.useProbeing_ = false;  // Using probeing technique.
			//parameter.label_ = ...;  // Initial configuration for improving.
			QPBO qpbo(gm, parameter);

			run_inference_algorithm<GraphicalModel>(gm.numberOfVariables(), qpbo, problem_name + inf_name);

			std::vector<bool> optimalVariables;
			const double partialOpt = qpbo.partialOptimality(optimalVariables);
			std::cout << "partial optimality: " << partialOpt << std::endl;
		}
	}

	// Move making algorithms.
	{
		std::cout << "\nExpansion-move algorithm --------------------------------------------" << std::endl;
		if (runExpansionMove)
		{
			const std::string inf_name("_expansion");

			// Set up the optimizer.
#if 1
			typedef opengm::external::MinSTCutKolmogorov<std::size_t, double> MinStCut;
#else
			typedef opengm::MinSTCutBoost<std::size_t, long, opengm::PUSH_RELABEL> MinStCut;
#endif
			typedef opengm::GraphCut<GraphicalModel, opengm::Minimizer, MinStCut> InferenceEngine;
			typedef opengm::AlphaExpansion<GraphicalModel, InferenceEngine> MinAlphaExpansion;

			const typename InferenceEngine::ValueType scale = 1000000;
			const typename InferenceEngine::Parameter infParameter(scale);
			const std::size_t maxNumberOfSteps = MAX_ITERATIONS;
			const typename MinAlphaExpansion::Parameter parameter(maxNumberOfSteps, infParameter);
			MinAlphaExpansion expansion(gm, parameter);

			run_inference_algorithm<GraphicalModel>(gm.numberOfVariables(), expansion, problem_name + inf_name);
		}

		std::cout << "\nSwap-move algorithm -------------------------------------------------" << std::endl;
		if (runSwapMove)
		{
			const std::string inf_name("_swap");

			// Set up the optimizer.
#if 1
			typedef opengm::external::MinSTCutKolmogorov<std::size_t, double> MinStCut;
#else
			typedef opengm::MinSTCutBoost<std::size_t, long, opengm::PUSH_RELABEL> MinStCut;
#endif
			typedef opengm::GraphCut<GraphicalModel, opengm::Minimizer, MinStCut> InferenceEngine;
			typedef opengm::AlphaBetaSwap<GraphicalModel, InferenceEngine> MinAlphaBetaSwap;

			const typename InferenceEngine::ValueType scale = 1000000;
			const typename InferenceEngine::Parameter infParameter(scale);
			typename MinAlphaBetaSwap::Parameter parameter;
			parameter.parameter_ = infParameter;
			parameter.maxNumberOfIterations_ = MAX_ITERATIONS;
			MinAlphaBetaSwap swap(gm, parameter);

			run_inference_algorithm<GraphicalModel>(gm.numberOfVariables(), swap, problem_name + inf_name);
		}

		std::cout << "\nIterative conditional modes (ICM) algorithm -------------------------" << std::endl;
		if (runICM)
		{
			const std::string inf_name("_icm");

			// Set up the optimizer.
			typedef opengm::ICM<GraphicalModel, opengm::Minimizer> ICM;

			//const ICM::MoveType moveType = ICM::SINGLE_VARIABLE;
			const std::vector<typename ICM::LabelType> startPoint;
			const typename ICM::Parameter parameter(startPoint);
			ICM icm(gm, parameter);

			run_inference_algorithm<GraphicalModel>(gm.numberOfVariables(), icm, problem_name + inf_name);
		}

		std::cout << "\nLazy flipper algorithm ----------------------------------------------" << std::endl;
		if (runLazyFlipper)
		{
			const std::string inf_name("_lf");

			// Set up the optimizer.
			typedef opengm::LazyFlipper<GraphicalModel, opengm::Minimizer> LazyFlipper;

			const std::size_t maxSubgraphSize = 2;
			const std::vector<typename LazyFlipper::LabelType> startPoint;
			const opengm::Tribool inferMultilabel = opengm::Tribool::Maybe;
			const typename LazyFlipper::Parameter parameter(maxSubgraphSize, inferMultilabel);
			LazyFlipper lf(gm, parameter);

			run_inference_algorithm<GraphicalModel>(gm.numberOfVariables(), lf, problem_name + inf_name);
		}

		std::cout << "\nLOC algorithm (stochastic move making algorithm) --------------------" << std::endl;
		if (runLOC)
		{
			const std::string inf_name("_loc");

			// Set up the optimizer.
			typedef opengm::LOC<GraphicalModel, opengm::Minimizer> LOC;

			const std::string solver("ad3");
			const double phi = 0.5;
			const std::size_t maxRadius = 10;
			const std::size_t maxIterations = MAX_ITERATIONS;
			const std::size_t aStarThreshold = 5;
			const typename LOC::Parameter parameter(solver, phi, maxRadius, maxIterations, aStarThreshold);
			LOC loc(gm, parameter);

			// Set starting point.
			std::vector<typename LOC::LabelType> startingPoint;
			// Assume startingPoint has been filled with meaningful labels.
			//loc.setStartingPoint(startingPoint.begin());

			run_inference_algorithm<GraphicalModel>(gm.numberOfVariables(), loc, problem_name + inf_name);
		}
	}

	// Sampling algorithms.
	{
		std::cout << "\nGibbs sampling algorithm --------------------------------------------" << std::endl;
		if (runnGibbs)
		{
			throw std::runtime_error("Not yet implemented");
		}

		std::cout << "\nSwendsen-Wang sampling algorithm ------------------------------------" << std::endl;
		if (runSwendsenWang)
		{
			throw std::runtime_error("Not yet implemented");
		}
	}

	std::cout << "\nBrute-force algorithm -----------------------------------------------" << std::endl;
	if (runBruteforce)
	{
		const std::string inf_name("_bf");

		// Set up the optimizer.
		typedef opengm::Bruteforce<GraphicalModel, opengm::Minimizer> Bruteforce;

		Bruteforce bf(gm);

		run_inference_algorithm<GraphicalModel>(gm.numberOfVariables(), bf, problem_name + inf_name);
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_opengm {

void inference_algorithms()
{
	std::cout << "Image segmentation (Potts model) ------------------------------------" << std::endl;
	if (false)
	{
		// Build the model.
		std::cout << "Creating a graphical model... ---------------------------------------" << std::endl;
		local::GraphicalModelForPottsModel gmForPottsModel;
		if (local::createGraphicalModelForPottsModel(gmForPottsModel))
			local::inference_algorithms<local::GraphicalModelForPottsModel>(gmForPottsModel, "segmentation");
	}

	std::cout << "\nStereo matching -----------------------------------------------------" << std::endl;
	if (false)
	{
		// Build the model.
		std::cout << "Creating a graphical model... ---------------------------------------" << std::endl;
		local::GraphicalModelForStereoMatching gmForStereoMatching;
		if (local::createGraphicalModelForStereoMatching(gmForStereoMatching))
			local::inference_algorithms<local::GraphicalModelForStereoMatching>(gmForStereoMatching, "stereo");
	}

	std::cout << "\nImage restoration ---------------------------------------------------" << std::endl;
	if (true)
	{
		// Build the model.
		std::cout << "Creating a graphical model... ---------------------------------------" << std::endl;
		local::GraphicalModelForImageRestoration gmForImageRestoration;
		if (local::createGraphicalModelForImageRestoration(gmForImageRestoration))
			local::inference_algorithms<local::GraphicalModelForImageRestoration>(gmForImageRestoration, "restoration");
	}
}

}  // namespace my_opengm
