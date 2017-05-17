#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/inference/swendsenwang.hxx>
#include <vector>
#include <iostream>


namespace {
namespace local {

typedef opengm::SimpleDiscreteSpace<> Space;
typedef opengm::ExplicitFunction<double> ExplicitFunction;
typedef opengm::PottsFunction<double> PottsFunction;
typedef opengm::meta::TypeListGenerator<ExplicitFunction, PottsFunction>::type FunctionTypes;
typedef opengm::GraphicalModel<double, opengm::Multiplier, FunctionTypes, Space> GraphicalModel;
typedef opengm::SwendsenWang<GraphicalModel, opengm::Maximizer> SwendsenWang;

// Build a Markov Chain with 10 binary variables in which
// - the first variable is more likely to be labeled 1 than 0
// - neighboring variables are more likely to have similar labels than dissimilar.
void buildGraphicalModel(GraphicalModel &gm)
{
	const std::size_t numberOfVariables = 10;
	const std::size_t numberOfLabels = 2;
	Space space(numberOfVariables, numberOfLabels);
	gm = GraphicalModel(space);

	// Add 1st order function.
	{
		ExplicitFunction func1(&numberOfLabels, &numberOfLabels + 1);
		func1(0) = 0.2;
		func1(1) = 0.8;
		const GraphicalModel::FunctionIdentifier fid1 = gm.addFunction(func1);

		// Add 1st order factor (at first variable).
		const std::size_t variableIndices[] = { 0 };
		gm.addFactor(fid1, variableIndices, variableIndices + 1);
	}

	// Add 2nd order function.
	{
		const double probEqual = 0.7;
		const double probUnequal = 0.3;
		const PottsFunction func2(numberOfLabels, numberOfLabels, probEqual, probUnequal);
		const GraphicalModel::FunctionIdentifier fid2 = gm.addFunction(func2);

		// Add 2nd order factors.
		for (std::size_t j = 0; j < numberOfVariables - 1; ++j)
		{
			const std::size_t variableIndices[] = { j, j + 1 };
			gm.addFactor(fid2, variableIndices, variableIndices + 2);
		}
	}
}

// Use Swendsen-Wang sampling for the purpose of finding the most probable labeling.
void swendsenWangSamplingForOptimization(const GraphicalModel &gm)
{
	const std::size_t numberOfSamplingSteps = 1e4;
	const std::size_t numberOfBurnInSteps = 1e4;
	const SwendsenWang::Parameter parameter(numberOfSamplingSteps, numberOfBurnInSteps);
	SwendsenWang swendsenWang(gm, parameter);

	std::cout << "On inferring ..." << std::endl;
	swendsenWang.infer();

	std::vector<std::size_t> argmax;
	swendsenWang.arg(argmax);

	std::cout << "Most probable labeling sampled: (";
	for (std::size_t j = 0; j < argmax.size(); ++j)
		std::cout << argmax[j] << ", ";
	std::cout << ")" << std::endl;
}

// Use Swendsen-Wang sampling to estimate marginals.
void swendsenWangSamplingForMarginalEstimation(const GraphicalModel &gm)
{
	typedef opengm::SwendsenWangMarginalVisitor<SwendsenWang> MarginalVisitor;

	MarginalVisitor visitor(gm);

	// Extend the visitor to sample first order marginals.
	for (std::size_t j = 0; j < gm.numberOfVariables(); ++j)
		visitor.addMarginal(j);

	// Extend the visitor to sample certain second order marginals.
	for (std::size_t j = 0; j < gm.numberOfVariables() - 1; ++j)
	{
		const std::size_t variableIndices[] = {j, j + 1};
		visitor.addMarginal(variableIndices, variableIndices + 2);
	}

	// Sample.
	SwendsenWang swendsenWang(gm);

	std::cout << "On inferring ..." << std::endl;
	swendsenWang.infer(visitor);

	// Output sampled first order marginals.
	std::cout << "Sampled first order marginals:" << std::endl;
	for (std::size_t j = 0; j < gm.numberOfVariables(); ++j)
	{
		std::cout << "x" << j << ": ";
		for (std::size_t k = 0; k < 2; ++k)
		{
			const double p = static_cast<double>(visitor.marginal(j)(k)) / visitor.numberOfSamples();
			std::cout << p << ' ';
		}
		std::cout << std::endl;
	}

	// Output sampled second order marginals.
	std::cout << "Sampled second order marginals:" << std::endl;
	for (std::size_t j = gm.numberOfVariables(); j < visitor.numberOfMarginals(); ++j)
	{
		std::cout << "var" << visitor.marginal(j).variableIndex(0) << ", var" << visitor.marginal(j).variableIndex(1) << ": ";
		for (std::size_t x = 0; x < 2; ++x)
			for (std::size_t y = 0; y < 2; ++y)
			{
				const double p = static_cast<double>(visitor.marginal(j)(x, y)) / visitor.numberOfSamples();
				std::cout << p << ' ';
			}
		std::cout << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_opengm {

// REF [file] >> ${OPENGM_HOME}/src/examples/unsorted-examples/swendsenwang.cxx
void swendsenwang_example()
{
	local::GraphicalModel gm;

	local::buildGraphicalModel(gm);
	local::swendsenWangSamplingForOptimization(gm);
	local::swendsenWangSamplingForMarginalEstimation(gm);
}

}  // namespace my_opengm
