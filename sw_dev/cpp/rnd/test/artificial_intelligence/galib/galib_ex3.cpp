#define _CRT_SECURE_NO_WARNINGS
//#define GALIB_USE_STD_NAMESPACE 1
#include <ga/GASimpleGA.h>	// the header for the GA we'll use
#include <ga/GA2DBinStrGenome.h> // and the header for the genome we need
#include <ga/std_stream.h>
#include <boost/timer/timer.hpp>


namespace {
namespace local {

// For the objective function we compare the contents of the genome with the target.
// If a bit is set in the genome and it is also set in the target, then we add 1 to the score.
// If the bits do not match, we don't do anything.
float objective(GAGenome &c)
{
	GA2DBinaryStringGenome & genome = (GA2DBinaryStringGenome &)c;
	short **pattern = (short **)c.userData();

	float value = 0.0f;
	for (int i = 0; i < genome.width(); ++i)
		for (int j = 0; j < genome.height(); ++j)
			value += (float)(genome.gene(i, j) == pattern[i][j]);

	return value;
}

}  // namespace local
}  // unnamed namespace

namespace my_galib {

// [ref] ${GALIB_HOME}/examples/ex3.c
void ex3(int argc, char *argv[])
{
	std::cout << "Example 3" << std::endl << std::endl;
	std::cout << "This program reads in a data file then runs a simple GA whose" << std::endl;
	std::cout << "objective function tries to match the pattern of bits that are" << std::endl;
	std::cout << "in the data file." << std::endl << std::endl;

	// See if we've been given a seed to use (for testing purposes).
	// When you specify a random seed, the evolution will be exactly the same each time you use that seed number.

	for (int ii = 1; ii < argc; ii++)
		if (std::strcmp(argv[ii++], "seed") == 0) {
			GARandomSeed((unsigned int)std::atoi(argv[ii]));
		}

	// Set the default values of the parameters and declare the params variable.
	// We use the genetic algorithm's configure member to set up the parameter list so that it will parse for the appropriate arguments.
	// Notice that the params argument 'removes' from the argv list any arguments that it recognized (actually it just re-orders them and changes the value of argc). 
	// Once the GA's parameters are registered, we set some values that we want that are different than the GAlib defaults.  Then we parse the command line.

	GAParameterList params;
	GASimpleGA::registerDefaultParameters(params);
	params.set(gaNscoreFilename, "data/optimization/galib/bog3.dat");
	params.set(gaNflushFrequency, 50);
	params.set(gaNpMutation, 0.001);
	params.set(gaNpCrossover, 0.8);
	params.parse(argc, argv, gaFalse);

	char filename[128] = "data/optimization/galib/smiley.txt";
	int i, j;

	// Parse the command line for arguments.

	for (i = 1; i < argc; ++i)
	{
		if (std::strcmp("file", argv[i]) == 0 || std::strcmp("f", argv[i]) == 0)
		{
			if (++i >= argc)
			{
				std::cout << argv[0] << ": the file option needs a filename.\n";
				return;
			}
			else{
				std::sprintf(filename, argv[i]);
				continue;
			}
		}
		else if (std::strcmp("seed", argv[i]) == 0)
		{
			if (++i < argc) continue;
			continue;
		}
		else
		{
			std::cout << argv[0] << ":  unrecognized arguement: " << argv[i] << std::endl << std::endl;
			std::cout << "valid arguments include standard GAlib arguments plus:" << std::endl;
			std::cout << "  f\tfilename from which to read (" << filename << ")" << std::endl << std::endl;
			return;
		}
	}

	// Read in the pattern from the specified file.
	// File format is pretty simple:
	// two integers that give the height then width of the matrix, then the matrix of 1's and 0's (with whitespace inbetween).

	std::ifstream inStream(filename);
	if (!inStream)
	{
		std::cout << "Cannot open " << filename << " for input." << std::endl;
		return;
	}

	int height, width;
	inStream >> height >> width;

	short **target = new short*[width];
	for (i = 0; i < width; ++i)
		target[i] = new short[height];

	for (j = 0; j < height; ++j)
		for (i = 0; i < width; ++i)
			inStream >> target[i][j];

	inStream.close();

	// Print out the pattern to be sure we got the right one.

	std::cout << "input pattern:" << std::endl;
	for (j = 0; j < height; ++j)
	{
		for (i = 0; i < width; ++i)
			std::cout << (target[i][j] == 1 ? '*' : ' ') << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	// Now create the GA and run it.

	GA2DBinaryStringGenome genome(width, height, local::objective, (void *)target);
	GASimpleGA ga(genome);
	ga.parameters(params);

	{
		boost::timer::cpu_timer timer;
	
		ga.evolve();

		boost::timer::cpu_times const elapsed_times(timer.elapsed());
		std::cout << "elpased time : " << (elapsed_times.system + elapsed_times.user) << std::endl;
	}

	std::cout << "best of generation data are in '" << ga.scoreFilename() << "'" << std::endl;
	genome = ga.statistics().bestIndividual();
	std::cout << "the ga generated:" << std::endl;
	for (j = 0; j < height; ++j)
	{
		for (i = 0; i < width; ++i)
			std::cout << (genome.gene(i,j) == 1 ? '*' : ' ') << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	for (i = 0; i < width; ++i)
		delete target[i];
	delete [] target;
}

}  // namespace my_galib
