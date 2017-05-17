#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/external/libdai/bp.hxx>
#include <opengm/inference/external/libdai/tree_reweighted_bp.hxx>
#include <opengm/inference/external/libdai/double_loop_generalized_bp.hxx>
#include <opengm/inference/external/libdai/fractional_bp.hxx>
#include <opengm/inference/external/libdai/tree_expectation_propagation.hxx>
#include <opengm/inference/external/libdai/exact.hxx>
#include <opengm/inference/external/libdai/junction_tree.hxx>
#include <opengm/inference/external/libdai/gibbs.hxx>
#include <opengm/inference/external/libdai/mean_field.hxx>
#include <vector>


namespace {
namespace local {

typedef opengm::SimpleDiscreteSpace<std::size_t, std::size_t> Space;
typedef OPENGM_TYPELIST_2(opengm::ExplicitFunction<double>, opengm::PottsFunction<double>) FunctionTypelist;
typedef opengm::GraphicalModel<double, opengm::Adder, FunctionTypelist, Space> GraphicalModel;

// REF [file] >> ${OPENGM_HOME}/src/examples/unsorted-examples/markov-chain.cxx
// Construct a graphical model with 
// - addition as the operation (template parameter Adder)
// - support for Potts functions (template parameter PottsFunction<double>).
void buildGraphicalModel(GraphicalModel &gm)
{
	// Construct a label space with numOfVariables many variables, each having numOfLabels many labels.
	const std::size_t numOfVariables = 20; 
	const std::size_t numOfLabels = 5;
	gm = GraphicalModel(Space(numOfVariables, numOfLabels));

	// For each variable, add one 1st order functions and one 1st order factor.
	for (std::size_t v = 0; v < numOfVariables; ++v)
	{
		const std::size_t shape[] = { numOfLabels };
		opengm::ExplicitFunction<double> func1(shape, shape + 1);
		for (std::size_t state = 0; state < numOfLabels; ++state)
			func1(state) = static_cast<double>(std::rand()) / RAND_MAX;

		const GraphicalModel::FunctionIdentifier fid1 = gm.addFunction(func1);

		const std::size_t variableIndices[] = { v };
		gm.addFactor(fid1, variableIndices, variableIndices + 1);
	}

	{
		// Add one (!) 2nd order Potts function.
		opengm::PottsFunction<double> func2(numOfLabels, numOfLabels, 0.0, 0.3);
		const GraphicalModel::FunctionIdentifier fid2 = gm.addFunction(func2);

		// For each pair of consecutive variables, add one factor that refers to the Potts function.
		for (std::size_t v = 0; v < numOfVariables - 1; ++v)
		{
			const std::size_t variableIndices[] = { v, v + 1 };
			gm.addFactor(fid2, variableIndices, variableIndices + 2);
		}
	}
}

void belief_propagation()
{
	GraphicalModel gm;
	buildGraphicalModel(gm);

	// Set up the optimizer (BP).
	typedef opengm::external::libdai::Bp<GraphicalModel, opengm::Maximizer> Bp;

	const std::size_t maxIterations = 100;
	const double damping = 0.0;
	const double tolerance = 0.000001;
	// Bp::UpdateRule = PARALL | SEQFIX | SEQRND | SEQMAX
	const Bp::UpdateRule updateRule = Bp::PARALL;
	const std::size_t verbose = 0;
	const std::size_t verboseLevel = 0;
	const Bp::Parameter parameter(maxIterations, damping, tolerance, updateRule, verboseLevel);
	Bp bp(gm, parameter);

	// Optimize (approximately).
	std::cout << "On inferring ..." << std::endl;
	bp.infer();

	// Obtain the (approximate) argmax.
	std::vector<std::size_t> labeling(gm.numberOfVariables());
	bp.arg(labeling);

	for (std::size_t var = 0; var < gm.numberOfVariables(); ++var)
		std::cout << "var" << var << "=" << labeling[var] << std::endl;
}

void tree_reweighted_belief_propagation()
{
	GraphicalModel gm;
	buildGraphicalModel(gm);

	// Set up the optimizer (trbp).
	typedef opengm::external::libdai::TreeReweightedBp<GraphicalModel, opengm::Maximizer> Trbp;

	const std::size_t maxIterations = 100;
	const double damping = 0.0;
	const double tolerance = 0.000001;
	const std::size_t ntrees = 10;
	// Trbp::UpdateRule = PARALL | SEQFIX | SEQRND | SEQMAX
	const Trbp::UpdateRule updateRule= Trbp::PARALL;
	const std::size_t verbose = 0;
	const std::size_t verboseLevel = 0;
	const Trbp::Parameter parameter(maxIterations, damping, tolerance, ntrees, updateRule, verboseLevel);
	Trbp trbp(gm, parameter);

	// Optimize (approximately).
	std::cout << "On inferring ..." << std::endl;
	trbp.infer();

	// Obtain the (approximate) argmax.
	std::vector<std::size_t> labeling(gm.numberOfVariables());
	trbp.arg(labeling);

	for (std::size_t var = 0; var < gm.numberOfVariables(); ++var)
		std::cout << "var" << var << "=" << labeling[var] << std::endl;
}

void double_loop_generalized_belief_propagation()
{
	GraphicalModel gm;
	buildGraphicalModel(gm);

	// Set up the optimizer (double loop generalized BP).
	typedef opengm::external::libdai::DoubleLoopGeneralizedBP<GraphicalModel, opengm::Minimizer> DoubleLoopGeneralizedBP;

	const bool doubleLoop = 1;
	// DoubleLoopGeneralizedBP::Clusters = MIN | BETHE | DELTA | LOOP
	const DoubleLoopGeneralizedBP::Clusters clusters = DoubleLoopGeneralizedBP::BETHE;
	const std::size_t loopDepth = 3;
	// DoubleLoopGeneralizedBP::Init = UNIFORM | RANDOM
	const DoubleLoopGeneralizedBP::Init init = DoubleLoopGeneralizedBP::UNIFORM;
	const std::size_t maxIterations = 10000;
	const double tolerance = 1e-9;
	const std::size_t verboseLevel = 0;
	const DoubleLoopGeneralizedBP::Parameter parameter(doubleLoop, clusters, loopDepth, init, maxIterations, tolerance, verboseLevel);
	DoubleLoopGeneralizedBP gdlbp(gm, parameter);

	// Optimize (approximately).
	std::cout << "On inferring ..." << std::endl;
	gdlbp.infer();

	// Obtain the (approximate) argmin.
	std::vector<std::size_t> labeling(gm.numberOfVariables());
	gdlbp.arg(labeling);

	for (std::size_t var = 0; var < gm.numberOfVariables(); ++var)
		std::cout << "var" << var << "=" << labeling[var] << std::endl;
}

void fractional_belief_propagation()
{
	GraphicalModel gm;
	buildGraphicalModel(gm);

	// Set up the optimizer (fractional BP).
	typedef opengm::external::libdai::FractionalBp<GraphicalModel, opengm::Maximizer> FractionalBp;

	const std::size_t maxIterations = 100;
	const double damping = 0.0;
	const double tolerance = 0.000001;
	// FractionalBp::UpdateRule = PARALL | SEQFIX | SEQRND | SEQMAX
	const FractionalBp::UpdateRule updateRule = FractionalBp::PARALL;
	const std::size_t verbose = 0;
	const std::size_t verboseLevel = 0;
	const FractionalBp::Parameter parameter(maxIterations, damping, tolerance, updateRule, verboseLevel);
	FractionalBp fbp(gm, parameter);

	// Optimize (approximately).
	std::cout << "On inferring ..." << std::endl;
	fbp.infer();

	// Obtain the (approximate) argmax.
	std::vector<std::size_t> labeling(gm.numberOfVariables());
	fbp.arg(labeling);

	for (std::size_t var = 0; var < gm.numberOfVariables(); ++var)
		std::cout << "var" << var << "=" << labeling[var] << std::endl;
}

void tree_expectation_propagation()
{
	GraphicalModel gm;
	buildGraphicalModel(gm);

	// Set up the optimizer (tree expectation propagation).
	typedef opengm::external::libdai::TreeExpectationPropagation<GraphicalModel, opengm::Maximizer> TreeExpectationPropagation;

	// TreeExpectationPropagation::TreeEpType = ORG | ALT
	const TreeExpectationPropagation::TreeEpType treeEpType = TreeExpectationPropagation::ORG;
	const std::size_t maxIterations = 10000;
	const double tolerance = 1e-9;
	const std::size_t verboseLevel = 0;
	const TreeExpectationPropagation::Parameter parameter(treeEpType, maxIterations, tolerance, verboseLevel);
	TreeExpectationPropagation treeEp(gm, parameter);

	// Optimize (approximately).
	std::cout << "On inferring ..." << std::endl;
	treeEp.infer();

	// Obtain the (approximate) argmax.
	std::vector<std::size_t> labeling(gm.numberOfVariables());
	treeEp.arg(labeling);

	for (std::size_t var = 0; var < gm.numberOfVariables(); ++var)
		std::cout << "var" << var << "=" << labeling[var] << std::endl;
}

void exact_inference()
{
	GraphicalModel gm;
	buildGraphicalModel(gm);

	// Set up the optimizer (junction tree).
	typedef opengm::external::libdai::Exact<GraphicalModel, opengm::Minimizer> Exact;

	const std::size_t verboseLevel = 0;
	const Exact::Parameter parameter(verboseLevel);
	Exact exact(gm, parameter);

	// Optimize (to global optimum).
	std::cout << "On inferring ..." << std::endl;
	exact.infer();
	
	// Obtain the argmin.
	std::vector<std::size_t> labeling(gm.numberOfVariables());
	exact.arg(labeling);

	for (std::size_t var = 0; var < gm.numberOfVariables(); ++var)
		std::cout << "var" << var << "=" << labeling[var] << std::endl;
}

void junction_tree_inference()
{
	GraphicalModel gm;
	buildGraphicalModel(gm);

	// Set up the optimizer (exact).
	typedef opengm::external::libdai::JunctionTree<GraphicalModel, opengm::Minimizer> JunctionTree;

	// JunctionTree::UpdateRule = HUGIN | SHSH
	const JunctionTree::UpdateRule updateRule = JunctionTree::HUGIN;
	// JunctionTree::Heuristic = MINFILL | WEIGHTEDMINFILL | MINNEIGHBORS
	const JunctionTree::Heuristic heuristic = JunctionTree::MINFILL;
	const std::size_t verboseLevel = 0;
	const JunctionTree::Parameter parameter(updateRule, heuristic, verboseLevel);
	JunctionTree jt(gm, parameter);

	// Optimize (to global optimum).
	std::cout << "Oon inferring ..." << std::endl;
	jt.infer();

	// Obtain the argmin.
	std::vector<std::size_t> labeling(gm.numberOfVariables());
	jt.arg(labeling);

	for (std::size_t var = 0; var < gm.numberOfVariables(); ++var)
		std::cout << "var" << var << "=" << labeling[var] << std::endl;
}

void gibbs_sampling()
{
	GraphicalModel gm;
	buildGraphicalModel(gm);

	// Set up the optimizer (Gibbs sampler).
	typedef opengm::external::libdai::Gibbs<GraphicalModel, opengm::Minimizer> Gibbs;

	const std::size_t maxIterations = 10000;
	const std::size_t burnIn = 100;
	const std::size_t restart = 10000;
	const std::size_t verboseLevel = 0;
	const Gibbs::Parameter parameter(maxIterations, burnIn, restart, verboseLevel);
	Gibbs gibbs(gm, parameter);

	// Optimize.
	std::cout << "On inferring ..." << std::endl;
	gibbs.infer();

	// Obtain the argmin.
	std::vector<std::size_t> labeling(gm.numberOfVariables());
	gibbs.arg(labeling);

	for (std::size_t var = 0; var < gm.numberOfVariables(); ++var)
		std::cout << "var" << var << "=" << labeling[var] << std::endl;
}

void mean_field_inference()
{
	GraphicalModel gm;
	buildGraphicalModel(gm);

	// Set up the optimizer (mean field).
	typedef opengm::external::libdai::MeanField<GraphicalModel, opengm::Minimizer> MeanField;

	const std::size_t maxIterations = 10000;
	const double damping = 0.2;
	const double tolerance = 1e-9;
	// MeanField::UpdateRule = NAIVE | HARDSPIN
	const MeanField::UpdateRule updateRule = MeanField::NAIVE;
	// MeanField::Init = UNIFORM | RANDOM
	const MeanField::Init init = MeanField::UNIFORM;
	const std::size_t verboseLevel = 0;
	const MeanField::Parameter parameter(maxIterations, damping, tolerance, updateRule, init, verboseLevel);
	MeanField mf(gm, parameter);

	// Optimize (approximately).
	std::cout << "On inferring ..." << std::endl;
	mf.infer();

	// Obtain the (approximate) argmin.
	std::vector<std::size_t> labeling(gm.numberOfVariables());
	mf.arg(labeling);

	for (std::size_t var = 0; var < gm.numberOfVariables(); ++var)
		std::cout << "var" << var << "=" << labeling[var] << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_opengm {

void libdai_interface()
{
	std::cout << "libDAI belief propagation -------------------------------------------" << std::endl;
	local::belief_propagation();

	std::cout << "\nlibDAI tree reweighted belief propagation ---------------------------" << std::endl;
	local::tree_reweighted_belief_propagation();

	std::cout << "\nlibDAI double loop generalized belief propagation -------------------" << std::endl;
	//local::double_loop_generalized_belief_propagation();  // Run-time error.

	std::cout << "\nlibDAI fractional belief propagation --------------------------------" << std::endl;
	local::fractional_belief_propagation();

	std::cout << "\nlibDAI tree expectation propagation ---------------------------------" << std::endl;
	local::tree_expectation_propagation();

	std::cout << "\nlibDAI exact inference ----------------------------------------------" << std::endl;
	local::exact_inference();

	std::cout << "\nlibDAI junction tree inference --------------------------------------" << std::endl;
	local::junction_tree_inference();

	std::cout << "\nlibDAI Gibbs sampling -----------------------------------------------" << std::endl;
	local::gibbs_sampling();

	std::cout << "\nlibDAI mean field inference -----------------------------------------" << std::endl;
	//local::mean_field_inference();  // Run-time error.
}

}  // namespace my_opengm
