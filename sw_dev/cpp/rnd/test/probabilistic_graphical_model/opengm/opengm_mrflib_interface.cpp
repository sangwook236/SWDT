#define MRFLABELVALUE int
#define MRFENERGYVALUE double
#define MRFCOSTVALUE double

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/external/mrflib.hxx>
#include <iostream>
#include <cstdlib>


namespace {
namespace local {

typedef opengm::GraphicalModel<double, opengm::Adder> GraphicalModelType;
typedef opengm::external::MRFLIB<GraphicalModelType> MRFLIB;

// [ref] ${OPENGM_HOME}/src/examples/unsorted-examples/quick_start.cxx
// construct a graphical model with 
// - 5 variables with { 5, 5, 2, 2, 10 } labels
// - addition as the operation (template parameter Adder)
void buildGraphicalModel(GraphicalModelType &gm)
{
	const std::size_t numsOfLabels[] = { 5, 5, 2, 2, 10 };
	gm = GraphicalModelType(opengm::DiscreteSpace<>(numsOfLabels, numsOfLabels + 5));

	typedef opengm::ExplicitFunction<float> ExplicitFunction;
	typedef GraphicalModelType::FunctionIdentifier FunctionIdentifier;

	// add 1st order functions and factors to the model
	for (std::size_t variable = 0; variable < gm.numberOfVariables(); ++variable)
	{
		// construct 1st order function
		const std::size_t shape[] = { gm.numberOfLabels(variable) };
		ExplicitFunction func1(shape, shape + 1);
		for (std::size_t state = 0; state < gm.numberOfLabels(variable); ++state)
			func1(state) = float(std::rand()) / RAND_MAX;  // random toy data

		// add function
		const FunctionIdentifier fid1 = gm.addFunction(func1);

		// add factor
		const std::size_t variableIndex[] = { variable };
		gm.addFactor(fid1, variableIndex, variableIndex + 1);
	}

	// add 3rd order functions and factors to the model
	for (std::size_t variable1 = 0; variable1 < gm.numberOfVariables(); ++variable1)
		for (std::size_t variable2 = variable1 + 1; variable2 < gm.numberOfVariables(); ++variable2)
			for (std::size_t variable3 = variable2 + 1; variable3 < gm.numberOfVariables(); ++variable3)
			{
				const std::size_t shape[] = {
					gm.numberOfLabels(variable1),
					gm.numberOfLabels(variable2),
					gm.numberOfLabels(variable3)
				};

				// construct 3rd order function
				ExplicitFunction func2(shape, shape + 3);
				for(std::size_t state1 = 0; state1 < gm.numberOfLabels(variable1); ++state1)
					for(std::size_t state2 = 0; state2 < gm.numberOfLabels(variable2); ++state2)
						for(std::size_t state3 = 0; state3 < gm.numberOfLabels(variable3); ++state3)      
							func2(state1, state2, state3) = float(std::rand()) / RAND_MAX;  // random toy data

				const FunctionIdentifier fid2 = gm.addFunction(func2);

				// sequences of variable indices need to be (and in this case are) sorted
				const std::size_t variableIndexSequence[] = { variable1, variable2, variable3 };
				gm.addFactor(fid2, variableIndexSequence, variableIndexSequence + 3);
			}
}

void icm()
{
	GraphicalModelType gm;
	buildGraphicalModel(gm);

	const MRFLIB::Parameter::InferenceType inferenceType = MRFLIB::Parameter::ICM;
	const MRFLIB::Parameter::EnergyType energyType = MRFLIB::Parameter::WEIGHTEDTABLE;
	const std::size_t numOfIterations = 10;
	const MRFLIB::Parameter para(inferenceType, energyType, numOfIterations);
	MRFLIB mrf(gm, para);

	std::cout << "on inferring ..." << std::endl;
	mrf.infer();

	std::cout << "value: " << mrf.value() << std::endl;

	// obtain the argmin
	std::vector<std::size_t> labeling(gm.numberOfVariables());
	mrf.arg(labeling);

	for (std::size_t var = 0; var < gm.numberOfVariables(); ++var)
		std::cout << "var" << var << "=" << labeling[var] << std::endl;
}

void alpha_expansion()
{
	GraphicalModelType gm;
	buildGraphicalModel(gm);

	const MRFLIB::Parameter::InferenceType inferenceType = MRFLIB::Parameter::EXPANSION;
	const MRFLIB::Parameter::EnergyType energyType = MRFLIB::Parameter::TL1;
	const std::size_t numOfIterations = 10;
	const MRFLIB::Parameter para(inferenceType, energyType, numOfIterations);
	MRFLIB mrf(gm, para);

	std::cout << "on inferring ..." << std::endl;
	mrf.infer();

	std::cout << "value: " << mrf.value() << std::endl;
}

void alpha_beta_swap()
{
	GraphicalModelType gm;
	buildGraphicalModel(gm);

	const MRFLIB::Parameter::InferenceType inferenceType = MRFLIB::Parameter::SWAP;
	const MRFLIB::Parameter::EnergyType energyType = MRFLIB::Parameter::TL1;
	const std::size_t numOfIterations = 10;
	const MRFLIB::Parameter para(inferenceType, energyType, numOfIterations);
	MRFLIB mrf(gm, para);

	std::cout << "on inferring ..." << std::endl;
	mrf.infer();

	std::cout << "value: " << mrf.value() << std::endl;
}

void lbp()
{
	GraphicalModelType gm;
	buildGraphicalModel(gm);

	const MRFLIB::Parameter::InferenceType inferenceType = MRFLIB::Parameter::MAXPRODBP;
	const MRFLIB::Parameter::EnergyType energyType = MRFLIB::Parameter::TABLES;
	const std::size_t numOfIterations = 10;
	const MRFLIB::Parameter para(inferenceType, energyType, numOfIterations);
	MRFLIB mrf(gm, para);

	std::cout << "on inferring ..." << std::endl;
	mrf.infer();

	std::cout << "value: " << mrf.value() << std::endl;
}

void bp_s()
{
	GraphicalModelType gm;
	buildGraphicalModel(gm);

	const MRFLIB::Parameter::InferenceType inferenceType = MRFLIB::Parameter::BPS;
	const MRFLIB::Parameter::EnergyType energyType = MRFLIB::Parameter::VIEW;
	const std::size_t numOfIterations = 10;
	const MRFLIB::Parameter para(inferenceType, energyType, numOfIterations);
	MRFLIB mrf(gm, para);

	std::cout << "on inferring ..." << std::endl;
	mrf.infer();

	std::cout << "value: " << mrf.value() << std::endl;
}

void trw_s()
{
	GraphicalModelType gm;
	buildGraphicalModel(gm);

	const MRFLIB::Parameter::InferenceType inferenceType = MRFLIB::Parameter::TRWS;
	const MRFLIB::Parameter::EnergyType energyType = MRFLIB::Parameter::VIEW;
	const std::size_t numOfIterations = 10;
	MRFLIB::Parameter para(inferenceType, energyType, numOfIterations);
	para.trwsTolerance_ = 1;
	MRFLIB mrf(gm, para);

	std::cout << "on inferring ..." << std::endl;
	mrf.infer();

	std::cout << "value: " << mrf.value() << " bound: " << mrf.bound() << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_opengm {

void mrflib_interface()
{
	std::cout << "MRFLib ICM ----------------------------------------------------------" << std::endl;
	local::icm();

	std::cout << "\nMRFLib a-expansion --------------------------------------------------" << std::endl;
	local::alpha_expansion();

	std::cout << "\nMRFLib ab-swap ------------------------------------------------------" << std::endl;
	local::alpha_beta_swap();

	std::cout << "\nMRFLib loopy belief propagation -------------------------------------" << std::endl;
	local::lbp();

	std::cout << "\nMRFLib BP-S ---------------------------------------------------------" << std::endl;
	local::bp_s();

	std::cout << "\nMRFLib TRW-S --------------------------------------------------------" << std::endl;
	local::trw_s();
}

}  // namespace my_opengm
