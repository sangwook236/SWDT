//#define GALIB_USE_STD_NAMESPACE 1
#include <ga/ga.h>
#include <ga/std_stream.h>
#include <boost/timer/timer.hpp>


namespace {
namespace local {

// This objective function just tries to match the genome to the pattern in the user data.
float objective(GAGenome &c)
{
	GA2DBinaryStringGenome &genome = (GA2DBinaryStringGenome &)c;
	GA2DBinaryStringGenome *pattern = (GA2DBinaryStringGenome *)c.userData();

	float value = 0.0f;
	for (int i = 0; i < genome.width(); ++i)
		for (int j = 0; j < genome.height(); ++j)
			value += (float)(genome.gene(i, j) == pattern->gene(i, j));

	return value;
}

}  // namespace local
}  // unnamed namespace

namespace my_galib {

// [ref] ${GALIB_HOME}/examples/ex7.c
void ex7(int argc, char *argv[])
{
	std::cout << "Example 7" << std::endl << std::endl;
	std::cout << "This program reads in a data file then runs a steady-state GA" << std::endl;
	std::cout << "whose objective function tries to match the pattern of bits that" << std::endl;
	std::cout << "are in the data file." << std::endl << std::endl;

	// See if we've been given a seed to use (for testing purposes).
	// When you specify a random seed, the evolution will be exactly the same each time you use that seed number.

	for (int ii = 1; ii < argc; ii++)
		if (std::strcmp(argv[ii++], "seed") == 0) {
			GARandomSeed((unsigned int)std::atoi(argv[ii]));
		}

	// Set the default values of the parameters.

	GAParameterList params;
	GASteadyStateGA::registerDefaultParameters(params);
	params.set(gaNpopulationSize, 50);  // number of individuals in population
	params.set(gaNpCrossover, 0.8);  // likelihood of doing crossover
	params.set(gaNpMutation, 0.001);  // probability of mutation
	params.set(gaNnGenerations, 200);  // number of generations
	params.set(gaNscoreFrequency, 20);  // how often to record scores
	params.set(gaNflushFrequency, 50);  // how often to flush scores to file
	params.set(gaNscoreFilename, "data/optimization/galib/bog7.dat");
	params.parse(argc, argv, gaFalse);

	char datafile[128] = "data/optimization/galib/smiley.txt";
	char parmfile[128] = "";
	int i, j;

	// Parse the command line for arguments.  We look for two possible arguments (after the parameter list has grabbed everything it recognizes).
	// One is the name of a data file from which to read, the other is the name of a parameters file from which to read.
	// Notice that any parameters in the parameters file will override the defaults above AND any entered on the command line.

	for (i = 1; i < argc; ++i)
	{
		if (std::strcmp("dfile", argv[i]) == 0)
		{
			if (++i >= argc)
			{
				std::cout << argv[0] << ": the data file option needs a filename." << std::endl;
				return;
			}
			else
			{
				std::sprintf(datafile, argv[i]);
				continue;
			}
		}
		else if (std::strcmp("pfile", argv[i]) == 0)
		{
			if (++i >= argc)
			{
				std::cout << argv[0] << ": the parameters file option needs a filename." << std::endl;
				return;
			}
			else
			{
				std::sprintf(parmfile, argv[i]);
				params.read(parmfile);
				continue;
			}
		}
		else if (strcmp("seed", argv[i]) == 0)
		{
			if (++i < argc) continue;
			continue;
		}
		else
		{
			std::cout << argv[0] << ":  unrecognized arguement: " << argv[i] << std::endl << std::endl;
			std::cout << "valid arguements include GAlib arguments plus:\n";
			std::cout << "  dfile\tdata file from which to read (" << datafile << ")" << std::endl;
			std::cout << "  pfile\tparameters file (" << parmfile << ")" << std::endl;
			std::cout << "default parameters are:\n" << params << std::endl << std::endl;
			return;
		}
	}

	// Read in the pattern from the specified file.
	// File format is pretty simple:
	// two integers that give the height then width of the matrix, then the matrix of 1's and 0's (with whitespace inbetween).
	// Here we use a binary string genome to store the desired pattern.
	// This shows how you can read in directly from a stream into a genome.
	// (This can be useful in a population initializer when you want to bias your population)

	std::ifstream inStream(datafile);
	if (!inStream)
	{
		std::cout << "Cannot open " << datafile << " for input." << std::endl;
		return;
	}

	int height, width;
	inStream >> height >> width;
	GA2DBinaryStringGenome target(width, height);
	inStream >> target;
	inStream.close();

	// Print out the pattern to be sure we got the right one.

	std::cout << "input pattern:" << std::endl;
	for (j = 0; j < height; ++j)
	{
		for (i = 0; i < width; ++i)
			std::cout << (target.gene(i,j) == 1 ? '*' : ' ') << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	// Now create the first genome and the GA.
	// When we create the genome, we give it not only the objective function but also 'user data'.
	// In this case, the user data is a pointer to our target pattern.
	// From a C++ point of view it would be better to derive a new genome class with its own data,
	// but here we just want a quick-and-dirty implementation, so we use the user-data.

	GA2DBinaryStringGenome genome(width, height, local::objective, (void *)&target);
	GASteadyStateGA ga(genome);

	// When you use a GA with overlapping populations, the default score frequency (how often the best of generation score is recorded) defaults to 100.
	// We use the parameters member function to change this value (along with all of the other parameters we set above).
	// You can also change the score frequency using the scoreFrequency member function of the GA.
	// Each of the parameters can be set individually if you like.
	// Here we just use the values that were set in the parameter list.

	ga.parameters(params);

	// The default selection method is RouletteWheel.
	// Here we set the selection method to TournamentSelection.

	GATournamentSelector selector;
	ga.selector(selector);

	// The following member functions override the values that were set using the parameter list.
	// They are commented out here so that you can see how they would be used.

	// We can control the amount of overlap from generation to generation using the pReplacement member function.
	// If we specify a value of 1 (100%) then the entire population is replaced each generation.
	// Notice that the percentage must be high enough to have at least one individual produced in each generation.
	// If not, the GA will post a warning message.

	//ga.pReplacement(0.3);

	// Often we use the number of generations as the criterion for terminating the GA run.
	// Here we override that and tell the GA to use convergence as a termination criterion.
	// Note that you can pass ANY function as the stopping criterion (as long as it has the correct signature).
	// Notice that the values we set here for p- and n-convergence override those that we set in the parameters object.

	ga.terminator(GAGeneticAlgorithm::TerminateUponConvergence);

	//ga.pConvergence(0.99);  // converge to within 1%
	//ga.nConvergence(100);  // within the last 100 generations

	// Evolve the GA 'by hand'.
	// When you use this method, be sure to initialize the GA before you start to evolve it.
	// You can print out the status of the current population by using the ga.population() member function.
	// This is also how you would print the status of the GA to disk along the way (rather than waiting to the end then printing the scores, for example).

	{
		boost::timer::cpu_timer timer;

		ga.initialize();
		while (!ga.done())
		{
			++ga;
		}
		ga.flushScores();

		boost::timer::cpu_times const elapsed_times(timer.elapsed());
		std::cout << "elpased time : " << (elapsed_times.system + elapsed_times.user) << std::endl;
	}

	// Now that the GA is finished, we set our default genome to equal the contents of the best genome that the GA found.
	// Then we print it out.

	genome = ga.statistics().bestIndividual();
	std::cout << "the ga generated:" << std::endl;
	for (j = 0; j < height; ++j)
	{
		for (i = 0; i < width; ++i)
			std::cout << (genome.gene(i,j) == 1 ? '*' : ' ') << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "best of generation data are in '" << ga.scoreFilename() << "'" << std::endl;
}

}  // namespace my_galib
