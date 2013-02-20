#include <opengm/opengm.hxx>
#include <opengm/datastructures/marray/marray.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/astar.hxx>
#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <string>


namespace {
namespace local {

template<class T>
void createAndPrintData(const std::size_t numOfVariables, marray::Marray<T> &data)
{
	const std::size_t shape[] = { numOfVariables, numOfVariables };
	data.resize(shape, shape + 2);

	std::cout << "pariwise costs:" << std::endl;
	//std::srand(0);
	for (std::size_t v = 0; v < data.shape(0); ++v)
	{
		for (std::size_t s = 0; s < data.shape(0); ++s)
		{
			data(v, s) = static_cast<float>(std::rand() % 100) * 0.01;
			std::cout << std::left << std::setw(6) << std::setprecision(2) << data(v, s);
		}
		std::cout << std::endl;
	}
}

void printSolution(const std::vector<std::size_t> &solution)
{
	std::set<std::size_t> unique;
	std::cout << std::endl << "solution labels :" << std::endl;
	for (std::size_t v = 0; v < solution.size(); ++v)
		std::cout << std::left << std::setw(2) << v << "  ->   " << solution[v] << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_opengm {

// [ref] ${OPENGM_HOME}/src/examples/unsorted-examples/one_to_one_matching.cxx
// [ref] "Efficient Belief Propagation for Early Vision", P. F. Felzenszwalb & D. Huttenlocher, IJCV, 2006
void bipartite_matching_example()
{
	// model parameters
	const std::size_t numOfVariables = 5;
	const std::size_t numOfLabels = numOfVariables;
	std::cout << std::endl << "matching with one to one correspondences:" << std::endl
		<< numOfVariables << " variables with " << numOfLabels <<" labels" << std::endl << std::endl;

	// pairwise costs
	marray::Marray<float> data;
	local::createAndPrintData(numOfVariables, data);

	// build the model with
	// - addition as the operation (template parameter Adder)
	// - support for Potts functions (template parameter PottsFunction<double>))
	// - numOfVariables variables, each having numOfLabels labels
	typedef opengm::ExplicitFunction<float> ExplicitFunction;
	typedef opengm::PottsFunction<float> PottsFunction;
	typedef opengm::GraphicalModel<
		float,
		opengm::Adder, 
		OPENGM_TYPELIST_2(PottsFunction, ExplicitFunction),
		opengm::SimpleDiscreteSpace<>
	> Model;
	typedef Model::FunctionIdentifier FunctionIdentifier;

	Model gm(opengm::SimpleDiscreteSpace<>(numOfVariables, numOfLabels));

	// add 1st order functions and factors
	{
		const std::size_t shape[] = { numOfLabels };
		ExplicitFunction func1(shape, shape + 1);
		for (std::size_t v = 0; v < numOfVariables; ++v)
		{
			for (std::size_t s = 0; s < numOfLabels; ++s)
				func1(s) = 1.0f - data(v, s);
			const FunctionIdentifier fid1 = gm.addFunction(func1);

			const std::size_t vi[] = { v };
			gm.addFactor(fid1, vi, vi + 1);
		}
	}

	// add 2nd order functions and factors
	{
		const float high = 20;

		// add one (!) 2nd order Potts function
		PottsFunction func2(numOfLabels, numOfLabels, high, 0);
		const FunctionIdentifier fid2 = gm.addFunction(func2);

		// add pair potentials for all variables
		for (std::size_t v1 = 0; v1 < numOfVariables; ++v1)
			for (std::size_t v2 = v1 + 1; v2 < numOfVariables; ++v2)
			{
				const std::size_t vi[] = { v1, v2 };
				gm.addFactor(fid2, vi, vi + 2);
			}
	}

	// set up the optimizer (A-star search)
	typedef opengm::AStar<Model, opengm::Minimizer> AstarType;
	AstarType astar(gm);

	// obtain and print the argmin
	AstarType::VerboseVisitorType verboseVisitor;

	std::cout << "\nA-star search:" << std::endl;
	astar.infer(verboseVisitor);

	std::vector<std::size_t> argmin;
	astar.arg(argmin);

	local::printSolution(argmin);
}

}  // namespace my_opengm
