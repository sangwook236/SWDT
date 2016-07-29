#define _CRT_SECURE_NO_WARNINGS
//#define GALIB_USE_STD_NAMESPACE 1
#include <ga/GASimpleGA.h>	// use the simple GA
#include <ga/GA2DBinStrGenome.h>  // use the 2D binary string genome
#include <ga/std_stream.h>
#include <boost/timer/timer.hpp>


namespace {
namespace local {

// This is the objective function.  All it does is check for alternating 0s and 1s.
// If the gene is odd and contains a 1, the fitness is incremented by 1.
// If the gene is even and contains a 0, the fitness is incremented by 1.
// No penalties are assigned.
// We have to do the cast because a plain, generic GAGenome doesn't have the members that a GA2DBinaryStringGenome has.
// And it's ok to cast it because we know that we will only get GA2DBinaryStringGenomes and nothing else.

float objective(GAGenome &g)
{
	GA2DBinaryStringGenome &genome = (GA2DBinaryStringGenome &)g;

	float score = 0.0f;
	int count = 0;

	for (int i = 0; i < genome.width(); ++i)
	{
		for (int j = 0; j < genome.height(); ++j)
		{
			if (genome.gene(i ,j) == 0 && count % 2 == 0)
				score += 1.0;
			if (genome.gene(i, j) == 1 && count % 2 != 0)
				score += 1.0;
			++count;
		}
	}
	return score;
}

}  // namespace local
}  // unnamed namespace

namespace my_galib {

// REF [file] >> ${GALIB_HOME}/examples/ex1.c
void ex1(int argc, char *argv[])
{
	std::cout << "Example 1\n" << std::endl;
	std::cout << "This program tries to fill a 2DBinaryStringGenome with" << std::endl;
	std::cout << "alternating 1s and 0s using a SimpleGA" << std::endl << std::endl;

	// See if we've been given a seed to use (for testing purposes).
	// When you specify a random seed, the evolution will be exactly the same each time you use that seed number.

	for (int ii = 1; ii < argc; ii++)
		if (std::strcmp(argv[ii++], "seed") == 0) {
			GARandomSeed((unsigned int)std::atoi(argv[ii]));
		}

	// Declare variables for the GA parameters and set them to some default values.
	const int width = 10;
	const int height = 5;
	const int popsize = 30;
	const int ngen = 400;
	const float pmut = 0.001f;
	const float pcross = 0.9f;

	// Now create the GA and run it.
	// First we create a genome of the type that we want to use in the GA.
	// The ga doesn't operate on this genome in the optimization - it just uses it to clone a population of genomes.

	GA2DBinaryStringGenome genome(width, height, local::objective);

	// Now that we have the genome, we create the genetic algorithm and set its parameters
	// - number of generations, mutation probability, and crossover probability.
	// And finally we tell it to evolve itself.

	GASimpleGA ga(genome);
	ga.populationSize(popsize);
	ga.nGenerations(ngen);
	ga.pMutation(pmut);
	ga.pCrossover(pcross);

	{
		boost::timer::cpu_timer timer;

		ga.evolve();

		boost::timer::cpu_times const elapsed_times(timer.elapsed());
		std::cout << "elpased time : " << (elapsed_times.system + elapsed_times.user) << std::endl;
	}

	// Now we print out the best genome that the GA found.

	std::cout << "The GA found:\n" << ga.statistics().bestIndividual() << std::endl;
}

}  // namespace my_galib
