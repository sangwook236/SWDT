#include <opengm/opengm.hxx>
#include <opengm/datastructures/marray/marray.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/lazyflipper.hxx>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>


namespace {
namespace local {

// this class is used to map a node (x, y) in the topological grid to a unique variable index
class TopologicalCoordinateToIndex
{
public:
	TopologicalCoordinateToIndex(const std::size_t geometricGridSizeX, const std::size_t geometricGridSizeY) 
	: gridSizeX_(geometricGridSizeX), gridSizeY_(geometricGridSizeY) 
	{}

	const std::size_t operator()(const std::size_t tx, const std::size_t ty) const
	{
		return tx / 2 + (ty / 2)*(gridSizeX_) + ((ty + ty % 2) / 2)*(gridSizeX_ - 1);
	}

	std::size_t gridSizeX_;
	std::size_t gridSizeY_;
};

template<class T>
void randomData(const std::size_t gridSizeX, const std::size_t gridSizeY, marray::Marray<T> &data)
{
	std::srand(gridSizeX * gridSizeY);
	const std::size_t shape[] = { gridSizeX, gridSizeY };
	data.assign();
	data.resize(shape, shape + 2);
	for (std::size_t y = 0; y < gridSizeY; ++y)
		for (std::size_t x = 0; x < gridSizeX; ++x)
			data(x, y) = static_cast<float>(std::rand() % 10) * 0.1f;
}

template<class T>
void printData(const marray::Marray<T> &data)
{
	std::cout << "energy for boundary to be active:" << std::endl;
	for (std::size_t y = 0; y < data.shape(1) * 2 - 1; ++y)
	{
		for (std::size_t x = 0; x < data.shape(0) * 2 - 1; ++x)
		{
			if (x % 2 == 0 && y % 2 == 0)
				std::cout << std::left << std::setw(3) << std::setprecision(1) << data(x / 2, y / 2);
			else if (x % 2 == 0 && y % 2 == 1)
				std::cout << std::left << std::setw(3) << std::setprecision(1) << "___";
			else if (x % 2 == 1 && y % 2 == 0)
				std::cout << std::left << std::setw(3) << std::setprecision(1) << " | ";
			else if (x % 2 == 1 && y % 2 == 1)
				std::cout << std::left << std::setw(3) << std::setprecision(1) << " * ";
		}
		std::cout << std::endl;
	}
}

// output the (approximate) argmin
template<class T>
void printSolution(const marray::Marray<T> &data, const std::vector<std::size_t> &solution) 
{
	TopologicalCoordinateToIndex cTHelper(data.shape(0), data.shape(1));
	std::cout << std::endl << "solution states:" << std::endl;
	std::cout << "solution:" << std::endl;
	for (std::size_t x = 0; x < data.shape(0) * 2 - 1; ++x)
		std::cout << std::left << std::setw(3) << std::setprecision(1) << "___";
	std::cout << std::endl;

	for (std::size_t y = 0; y < data.shape(1) * 2 - 1; ++y)
	{
		std::cout << "|";
		for (std::size_t x = 0; x < data.shape(0) * 2 - 1; ++x)
		{
			if (x % 2 == 0 && y % 2 == 0)
			{
				data(x / 2, y / 2) = static_cast<float>(std::rand() % 10) * 0.1f;
				std::cout << std::left << std::setw(3) << std::setprecision(1) << " ";
			}
			else if (x % 2 == 0 && y % 2 == 1)
			{
				if (solution[cTHelper(x, y)])
					std::cout << std::left << std::setw(3) << std::setprecision(1) << "___";
				else
					std::cout << std::left << std::setw(3) << std::setprecision(1) << "   ";
			}
			else if (x % 2 == 1 && y % 2 == 0)
			{
				if (solution[cTHelper(x, y)])
					std::cout << std::left << std::setw(3) << std::setprecision(1) << " | ";
				else
					std::cout << std::left << std::setw(3) << std::setprecision(1) << "   ";
			}
			else if (x % 2 == 1 && y % 2 == 1)
				std::cout << std::left << std::setw(3) << std::setprecision(1) << " * ";
		}
		std::cout << "|" << std::endl;
	}
	for (std::size_t x = 0; x < data.shape(0) * 2 - 1; ++x)
		std::cout << std::left << std::setw(3) << std::setprecision(1) << "___";
	std::cout << std::endl;
}

// user defined Function Type
template<class T>
struct ClosednessFunctor
{
public:
	typedef T value_type;

	template<class Iterator>
	inline const T operator()(Iterator begin) const
	{
		std::size_t sum = begin[0];
		sum += begin[1];
		sum += begin[2];
		sum += begin[3];
		if (sum != 2 && sum != 0)
			return high;
		return 0;
	}

	std::size_t dimension() const
	{
		return 4;
	}

	std::size_t shape(const std::size_t i) const
	{
		return 2;
	}

	std::size_t size() const
	{
		return 16;
	}

	T high;
};

}  // namespace local
}  // unnamed namespace

namespace my_opengm {

// [ref] ${OPENGM_HOME}/src/examples/image-processing-examples/interpixel_boundary_segmentation.cxx
void interpixel_boundary_segmentation_example()
{
	// model parameters
	const std::size_t gridSizeX = 5, gridSizeY = 5;  // size of grid
	const float beta = 0.9;  // bias to choose between under- and over-segmentation
	const float high = 10;  // closedness-enforcing soft-constraint

	// size of the topological grid
	const std::size_t tGridSizeX = 2 * gridSizeX - 1, tGridSizeY = 2 * gridSizeY - 1;
	const std::size_t numOfVariables = gridSizeY * (gridSizeX - 1) + gridSizeX * (gridSizeY - 1);
	const std::size_t dimT[] = { tGridSizeX, tGridSizeY };
	local::TopologicalCoordinateToIndex cTHelper(gridSizeX, gridSizeY);
	marray::Marray<float> data;
	local::randomData(gridSizeX, gridSizeY, data);

	std::cout << "interpixel boundary segmentation with closedness:" << std::endl;
	local::printData(data);

	// construct a graphical model with 
	// - addition as the operation (template parameter Adder)
	// - the user defined function type ClosednessFunctor<float>
	// - gridSizeY * (gridSizeX - 1) + gridSizeX * (gridSizeY - 1) variables, each having 2 many labels.
	typedef opengm::meta::TypeListGenerator<
		opengm::ExplicitFunction<float>,
		local::ClosednessFunctor<float>
	>::type FunctionTypeList;
	typedef opengm::GraphicalModel<float, opengm::Adder, FunctionTypeList, opengm::SimpleDiscreteSpace<> > Model;
	typedef Model::FunctionIdentifier FunctionIdentifier;
	Model gm(opengm::SimpleDiscreteSpace<>(numOfVariables, 2));

	// for each boundary in the grid, i.e. for each variable of the model, add one 1st order functions and one 1st order factor
	{
		const std::size_t shape[] = { 2 };
		opengm::ExplicitFunction<float> func1(shape, shape + 1);
		for (std::size_t yT = 0; yT < dimT[1]; ++yT)
			for (std::size_t xT = 0; xT < dimT[0]; ++xT)
				if ((xT % 2 + yT % 2) == 1)
				{
					const float gradient = std::fabs(data(xT / 2, yT / 2) - data(xT / 2 + xT % 2, yT / 2 + yT % 2));              
					func1(0) = beta * gradient;  // value for inactive boundary               
					func1(1) = (1.0 - beta) * (1.0 - gradient);  // value for active boundary
					const FunctionIdentifier fid1 = gm.addFunction(func1);

					const std::size_t vi[] = { cTHelper(xT, yT) };
					gm.addFactor(fid1, vi, vi + 1);
				}
	}

	// for each junction of four inter-pixel edges on the grid, 
	// one factor is added that connects the corresponding variable indices and refers to the ClosednessFunctor function
	{
		// add one (!) 4th order ClosednessFunctor function
		local::ClosednessFunctor<float> func4;
		func4.high = high;
		const FunctionIdentifier fid4 = gm.addFunction(func4);

		// add factors
		for (std::size_t y = 0; y < dimT[1]; ++y)
			for (std::size_t x = 0; x < dimT[0]; ++x)
				if (x % 2 + y % 2 == 2)
				{
					std::size_t vi[] = {
						cTHelper(x + 1, y),
						cTHelper(x - 1, y),
						cTHelper(x, y + 1),
						cTHelper(x, y - 1)
					};
					std::sort(vi, vi + 4);
					gm.addFactor(fid4, vi, vi + 4);
				}
	}

	// set up the optimizer (lazy flipper)
	typedef opengm::LazyFlipper<Model, opengm::Minimizer> LazyFlipperType;
	LazyFlipperType::VerboseVisitorType verboseVisitor;
	const std::size_t maxSubgraphSize = 5;
	LazyFlipperType lazyFlipper(gm, maxSubgraphSize);

	// obtain the (approximate) argmin
	std::cout << "start inference:" << std::endl;
	lazyFlipper.infer(verboseVisitor);

	// output the (approximate) argmin
	std::vector<std::size_t> solution;
	lazyFlipper.arg(solution);
	local::printSolution(data, solution);
}

}  // namespace my_opengm
