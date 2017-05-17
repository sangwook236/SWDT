#include <opengm/opengm.hxx>
#include <opengm/datastructures/marray/marray.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <algorithm>
#include <iostream>


namespace {
namespace local {

// This function maps a node (x, y) in the grid to a unique variable index.
inline std::size_t variableIndex(const std::size_t Nx, const std::size_t x, const std::size_t y)
{ 
	return x + Nx * y;
}

}  // namespace local
}  // unnamed namespace

namespace my_opengm {

// REF [file] >> ${OPENGM_HOME}/src/examples/image-processing-examples/grid_potts.cxx
// REF [paper] >> "Efficient Belief Propagation for Early Vision", P. F. Felzenszwalb & D. Huttenlocher, IJCV, 2006
void potts_model_on_2d_grid_example()
{
	// Model parameters (global variables are used only in example code).
	const std::size_t Nx = 30;  // Width of the grid.
	const std::size_t Ny = 30;  // Height of the grid.
	const std::size_t numOfLabels = 5;
	const double lambda = 0.1;  // Coupling strength of the Potts model.

	// Construct a label space with
	// - Nx * Ny variables.
	// - each having numOfLabels many labels.
	typedef opengm::SimpleDiscreteSpace<std::size_t, std::size_t> Space;
	Space space(Nx * Ny, numOfLabels);

	// Construct a graphical model with 
	// - addition as the operation (template parameter Adder).
	// - support for Potts functions (template parameter PottsFunction<double>).
	typedef opengm::GraphicalModel<double, opengm::Adder, OPENGM_TYPELIST_2(opengm::ExplicitFunction<double>, opengm::PottsFunction<double>), Space> Model;
	Model gm(space);

	// Local model.
	{
		// For each node (x, y) in the grid, i.e. for each variable variableIndex(Nx, x, y) of the model,
		// add one 1st order functions and one 1st order factor.
		for (std::size_t y = 0; y < Ny; ++y) 
			for (std::size_t x = 0; x < Nx; ++x)
			{
				// Function.
				const std::size_t shape[] = { numOfLabels };
				opengm::ExplicitFunction<double> func1(shape, shape + 1);
				for (std::size_t state = 0; state < numOfLabels; ++state)
					func1(state) = (1.0 - lambda) * std::rand() / RAND_MAX;

				const Model::FunctionIdentifier fid1 = gm.addFunction(func1);

				// Factor.
				const std::size_t variableIndices[] = { local::variableIndex(Nx, x, y) };
				gm.addFactor(fid1, variableIndices, variableIndices + 1);
			}
	}

	// Pairwise interaction.
	{
		// Add one (!) 2nd order Potts function.
		opengm::PottsFunction<double> func2(numOfLabels, numOfLabels, 0.0, lambda);
		const Model::FunctionIdentifier fid2 = gm.addFunction(func2);

		// For each pair of nodes (x1, y1), (x2, y2) which are adjacent on the grid,
		// add one factor that connects the corresponding variable indices and refers to the Potts function.
		for (std::size_t y = 0; y < Ny; ++y) 
			for (std::size_t x = 0; x < Nx; ++x)
			{
				if (x + 1 < Nx)  // (x, y) -- (x + 1, y).
				{
					std::size_t variableIndices[] = { local::variableIndex(Nx, x, y), local::variableIndex(Nx, x + 1, y) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
				if (y + 1 < Ny)  // (x, y) -- (x, y + 1).
				{
					std::size_t variableIndices[] = { local::variableIndex(Nx, x, y), local::variableIndex(Nx, x, y + 1) };
					std::sort(variableIndices, variableIndices + 2);
					gm.addFactor(fid2, variableIndices, variableIndices + 2);
				}
			}
	}

	// Set up the optimizer (loopy belief propagation).
	typedef opengm::BeliefPropagationUpdateRules<Model, opengm::Minimizer> UpdateRules;
	typedef opengm::MessagePassing<Model, opengm::Minimizer, UpdateRules, opengm::MaxDistance> BeliefPropagation;

	const std::size_t maxNumberOfIterations = 40;
	const double convergenceBound = 1e-7;
	const double damping = 0.5;
	const BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
	BeliefPropagation bp(gm, parameter);

	// Optimize (approximately).
	std::cout << "on inferring ..." << std::endl;
	BeliefPropagation::VerboseVisitorType visitor;
	bp.infer(visitor);

	// Obtain the (approximate) argmin.
	std::vector<std::size_t> labeling(Nx * Ny);
	bp.arg(labeling);

	// Output the (approximate) argmin.
	std::size_t variableIndex = 0;
	for (std::size_t y = 0; y < Ny; ++y)
	{
		for (std::size_t x = 0; x < Nx; ++x)
		{
			std::cout << labeling[variableIndex] << ' ';
			++variableIndex;
		}   
		std::cout << std::endl;
	}
}

}  // namespace my_opengm
