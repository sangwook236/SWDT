#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/graphicalmodel/space/discretespace.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/inference/icm.hxx>
#include <opengm/inference/gibbs.hxx>
#include <iostream>
#include <vector>


namespace {
namespace local {

void basic_operation()
{
	// [ref] ${OPENGM_HOME}/src/examples/unsorted-examples/space_types.cxx
	{
		typedef float ValueType;
		typedef opengm::UInt32Type IndexType;
		typedef opengm::UInt8Type LabelType;
		typedef opengm::Adder OperationType;
		typedef opengm::ExplicitFunction<ValueType> FunctionType;

		// dense space where all variables can have a different number of variables
		{
			typedef opengm::DiscreteSpace<IndexType, LabelType> SpaceType;
			typedef opengm::GraphicalModel<
				ValueType,
				OperationType,
				FunctionType,
				SpaceType
			> GraphicalModelType;
			const LabelType numsOfLabels[] = { 2, 4, 6, 4, 3 };

			// graphical model with 5 variables with 2, 4, 6, 4, and 3 labels
			GraphicalModelType gm(SpaceType(numsOfLabels, numsOfLabels + 5));
		}

		// simple space where all variables have the same number of labels
		{
			typedef opengm::SimpleDiscreteSpace<IndexType, LabelType> SpaceType;
			typedef opengm::GraphicalModel<
				ValueType,
				OperationType,
				FunctionType,
				SpaceType
			> GraphicalModelType;

			// graphical model with 5 variables, each having 2 labels 
			const IndexType numOfVariables = 5;
			const LabelType numOfLabels = 4;
			GraphicalModelType gm(SpaceType(numOfVariables, numOfLabels));
		}
	}

	// [ref] ${OPENGM_HOME}/src/examples/unsorted-examples/inference_types.cxx
	{
		// construct a graphical model with 
		// - addition as the operation (template parameter Adder)
		// - support for Potts functions (template parameter PottsFunction<double>)
		typedef OPENGM_TYPELIST_2(opengm::ExplicitFunction<double>, opengm::PottsFunction<double>) FunctionTypelist;
		typedef opengm::SimpleDiscreteSpace<std::size_t, std::size_t> Space;
		typedef opengm::GraphicalModel<double, opengm::Adder, FunctionTypelist, Space> Model;
		Model gm;

		// Let us assume we have added some variables, factors and functions to the graphical model gm
		// (see other exampels for building a model)
		{
			// [ref] ${OPENGM_HOME}/src/examples/unsorted-examples/markov-chain.cxx

			const std::size_t numberOfVariables = 10;
			const std::size_t numberOfLabels = 2;
			gm = Model(Space(numberOfVariables, numberOfLabels));

			// for each variable, add one 1st order functions and one 1st order factor
			for (std::size_t v = 0; v < numberOfVariables; ++v)
			{
				const std::size_t shape[] = { numberOfLabels };
				opengm::ExplicitFunction<double> func1(shape, shape + 1);
				for (std::size_t state = 0; state < numberOfLabels; ++state)
					func1(state) = static_cast<double>(std::rand()) / RAND_MAX;

				const Model::FunctionIdentifier fid1 = gm.addFunction(func1);

				const std::size_t variableIndices[] = { v };
				gm.addFactor(fid1, variableIndices, variableIndices + 1);
			}

			{
				// add one (!) 2nd order Potts function
				opengm::PottsFunction<double> func2(numberOfLabels, numberOfLabels, 0.0, 0.3);
				const Model::FunctionIdentifier fid2 = gm.addFunction(func2);

				// for each pair of consecutive variables, add one factor that refers to the Potts function
				for (std::size_t v = 0; v < numberOfVariables - 1; ++v)
				{
					const std::size_t variableIndices[] = { v, v + 1 };
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
			}
		}

		// Optimize the model with different Optimizers

		// Gibbs
		{
			// typedefs to a Gibbs minimizer and maximizer
			typedef opengm::Gibbs<Model, opengm::Minimizer> OptimizerMinimizerType;
			typedef opengm::Gibbs<Model, opengm::Maximizer> OptimizerMaximizerType;
			typedef OptimizerMinimizerType::Parameter OptimizerMinimizerParameterType;
			typedef OptimizerMaximizerType::Parameter OptimizerMaximizerParameterType;

			// construct solver parameters (all parameters have default values)
			const OptimizerMinimizerParameterType minimizerParameter(
				1000000,   // number of iterations
				100000  // stop after 100000 iterations without improvement
			);
			// default parameter
			const OptimizerMaximizerParameterType maximizerParameter;

			// construct optimizers (minimizer and maximizer)
			OptimizerMinimizerType optimizerMinimizer(gm, minimizerParameter);
			OptimizerMaximizerType optimizerMaximizer(gm, maximizerParameter);

			// optimize the models (minimizer and maximizer)
			std::cout << "on inferring ..." << std::endl;
			optimizerMinimizer.infer();
			optimizerMaximizer.infer();

			// get the argmin / argmax 
			std::vector<Model::LabelType> argmin, argmax;
			optimizerMinimizer.arg(argmin);
			optimizerMaximizer.arg(argmax);
		}

		// ICM
		{
			// typedefs to a ICM minimizer and maximizer
			typedef opengm::ICM<Model, opengm::Minimizer> OptimizerMinimizerType;
			typedef opengm::ICM<Model, opengm::Maximizer> OptimizerMaximizerType;
			typedef OptimizerMinimizerType::Parameter OptimizerMinimizerParameterType;
			typedef OptimizerMaximizerType::Parameter OptimizerMaximizerParameterType;

			// construct solver parameters (all parameters have default values)
			std::vector<Model::LabelType> startingPoint(gm.numberOfVariables());

			//fill starting point
			// ....

			// assume starting point is filled with labels
			OptimizerMinimizerParameterType minimizerParameter(
				OptimizerMinimizerType::SINGLE_VARIABLE,  // flip a single variable (FACTOR for flip all var. a factor depends on)
				startingPoint
			);

			// without starting point
			OptimizerMaximizerParameterType maximizerParameter(
				OptimizerMaximizerType::FACTOR,  // flip a single variable (FACTOR for flip all var. a factor depends on)
				startingPoint
			);

			// construct optimizers (minimizer and maximizer)
			OptimizerMinimizerType optimizerMinimizer(gm,minimizerParameter);
			OptimizerMaximizerType optimizerMaximizer(gm,maximizerParameter);

			// optimize the models (minimizer and maximizer)
			std::cout << "on inferring ..." << std::endl;
			optimizerMinimizer.infer();
			optimizerMaximizer.infer();

			// get the argmin / argmax 
			std::vector<Model::LabelType> argmin, argmax;
			optimizerMinimizer.arg(argmin);
			optimizerMaximizer.arg(argmax);
		}
	}
}

// [ref] ${OPENGM_HOME}/src/examples/io-examples/io_graphical_model.cxx
void input_output_example()
{
	typedef opengm::GraphicalModel<float, opengm::Multiplier> GraphicalModel;

	// build a graphical model (other examples have more details)
	const std::size_t numsOfLabels[] = { 3, 3, 3, 3 };
	GraphicalModel gmA(opengm::DiscreteSpace<>(numsOfLabels, numsOfLabels + 4)); 

	const std::size_t shape[] = { 3 };
	opengm::ExplicitFunction<float> func(shape, shape + 1); 
	for (std::size_t i = 0; i < gmA.numberOfVariables(); ++i)
	{
		const std::size_t vi[] = { i };
		func(0) = float(i);
		func(1) = float(i + 1);
		func(2) = float(i - 2);
		
		const GraphicalModel::FunctionIdentifier fid = gmA.addFunction(func);
		gmA.addFactor(fid, vi, vi + 1);
	}

	// save graphical model into an hdf5 dataset named "toy-gm"
	opengm::hdf5::save(gmA, "./probabilistic_graphical_model_data/opengm/gm.h5", "toy-gm");

	// load the graphical model from the hdf5 dataset
	GraphicalModel gmB;
	opengm::hdf5::load(gmB, "./probabilistic_graphical_model_data/opengm/gm.h5","toy-gm");
}

}  // namespace local
}  // unnamed namespace

namespace my_opengm {

void quick_start_example();
void markov_chain_example();
void gibbs_example();
void swendsenwang_example();
void bipartite_matching_example();
void potts_model_on_2d_grid_example();
void interpixel_boundary_segmentation_example();

void inference_algorithms();

void libdai_interface();
void mrflib_interface();

}  // namespace my_opengm

int opengm_main(int argc, char *argv[])
{
	std::cout << "basic operation -----------------------------------------------------" << std::endl;
	//local::basic_operation();

	// examples
	{
		std::cout << "\ninput & output example ----------------------------------------------" << std::endl;
		//local::input_output_example();  // HDF5

		std::cout << "\nquick start example -------------------------------------------------" << std::endl;
		//my_opengm::quick_start_example();  // 3rd order explicit function + ICM

		std::cout << "\nMarkov chain example ------------------------------------------------" << std::endl;
		//my_opengm::markov_chain_example();  // Potts model + BP

		std::cout << "\nGibbs sampling example ----------------------------------------------" << std::endl;
		//my_opengm::gibbs_example();  // Potts model + Gibbs sampling

		std::cout << "\nSwendsen-Wang sampling example --------------------------------------" << std::endl;
		//my_opengm::swendsenwang_example();  // Potts model + Swendsen-Wang sampling

		std::cout << "\nbipartite matching example ------------------------------------------" << std::endl;
		//my_opengm::bipartite_matching_example();  // Potts model + A-star search

		std::cout << "\nPotts model on a 2-dim grid example ---------------------------------" << std::endl;
		//my_opengm::potts_model_on_2d_grid_example();  // Potts model + BP

		std::cout << "\ninterpixel boundary segmentation example ----------------------------" << std::endl;
		my_opengm::interpixel_boundary_segmentation_example();  // 4th order closedness (user-defined) function + lazy flipper
	}

	// inference algorithm
	{
		// an example for segmentation
		//	[ref] ${CPP_RND_HOME}/test/segmentation/interactive_graph_cuts/interactive_graph_cuts_main.cpp

		std::cout << "\ninference algorithms ------------------------------------------------" << std::endl;
		// image segmentation, stereo matching, & image restoration
		//my_opengm::inference_algorithms();
	}

	// external library interfacing
	{
		std::cout << "\nlibDAI interface ----------------------------------------------------" << std::endl;
		//my_opengm::libdai_interface();

		std::cout << "\nMRFLib interface ----------------------------------------------------" << std::endl;
		//my_opengm::mrflib_interface();  // compile-time error
	}

	return 0;
}
