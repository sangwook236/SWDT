#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opengm {

// REF [file] >> ${OPENGM_HOME}/src/examples/unsorted-examples/markov-chain.cxx
void markov_chain_example()
{
	// Construct a label space with numOfVariables many variables, each having numOfLabels many labels.
	const std::size_t numOfVariables = 40; 
	const std::size_t numOfLabels = 5;

	typedef opengm::SimpleDiscreteSpace<std::size_t, std::size_t> Space;

	Space space(numOfVariables, numOfLabels);

	// Construct a graphical model with 
	// - addition as the operation (template parameter Adder).
	// - support for Potts functions (template parameter PottsFunction<double>).
	typedef OPENGM_TYPELIST_2(opengm::ExplicitFunction<double>, opengm::PottsFunction<double>) FunctionTypelist;
	typedef opengm::GraphicalModel<double, opengm::Adder, FunctionTypelist, Space> Model;

	Model gm(space);

	// Local model.
	// For each variable, add one 1st order functions and one 1st order factor.
	for (std::size_t v = 0; v < numOfVariables; ++v)
	{
		const std::size_t shape[] = { numOfLabels };
		opengm::ExplicitFunction<double> func1(shape, shape + 1);
		for (std::size_t state = 0; state < numOfLabels; ++state)
			func1(state) = static_cast<double>(std::rand()) / RAND_MAX;

		const Model::FunctionIdentifier fid1 = gm.addFunction(func1);

		const std::size_t variableIndices[] = { v };
		gm.addFactor(fid1, variableIndices, variableIndices + 1);
	}

	// Pairwise interaction.
	{
		// Add one (!) 2nd order Potts function.
		opengm::PottsFunction<double> func2(numOfLabels, numOfLabels, 0.0, 0.3);
		const Model::FunctionIdentifier fid2 = gm.addFunction(func2);

		// For each pair of consecutive variables, add one factor that refers to the Potts function.
		for (std::size_t v = 0; v < numOfVariables - 1; ++v)
		{
			const std::size_t variableIndices[] = { v, v + 1 };
			gm.addFactor(fid2, variableIndices, variableIndices + 2);
		}
	}

	// FIXME [check] >> why is this a markov chain example?

	// Set up the optimizer (loopy belief propagation).
	typedef opengm::BeliefPropagationUpdateRules<Model, opengm::Minimizer> UpdateRules;
	typedef opengm::MessagePassing<Model, opengm::Minimizer, UpdateRules, opengm::MaxDistance> BeliefPropagation;

	const std::size_t maxNumberOfIterations = numOfVariables * 2;
	const double convergenceBound = 1e-7;
	const double damping = 0.0;
	const BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
	BeliefPropagation bp(gm, parameter);

	// Optimize (approximately).
	BeliefPropagation::VerboseVisitorType visitor;

	std::cout << "On inferring ..." << std::endl;
	bp.infer(visitor);

	// Obtain the (approximate) argmin.
	std::vector<std::size_t> labeling(numOfVariables);
	bp.arg(labeling);

	for (std::size_t var = 0; var < gm.numberOfVariables(); ++var)
		std::cout << "var" << var << "=" << labeling[var] << std::endl;
}

}  // namespace my_opengm
