#include "simfunctions.h"
#include <smctc/smctc.hh>
#include <iostream>
#include <cstdio> 
#include <cstdlib>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_smctc {

// [ref] ${SMCTC_HOME}/examples/rare-events/main.cc
void rare_events_example()
{
	// Number of Particles
	const long lNumber = 1000;

	// An array of move function pointers
	void (*pfMoves[])(long, smc::particle<mChain<double> > &, smc::rng *) = { fMove1, fMove2 };

	smc::moveset<mChain<double> > Moveset(fInitialise, fSelect, sizeof(pfMoves) / sizeof(pfMoves[0]), pfMoves, fMCMC);
	smc::sampler<mChain<double> > Sampler(lNumber, SMC_HISTORY_RAM);

	Sampler.SetResampleParams(SMC_RESAMPLE_STRATIFIED, 0.5);
	Sampler.SetMoveSet(Moveset);

	Sampler.Initialise();
	Sampler.IterateUntil(lIterates);

	// Estimate the normalising constant of the terminal distribution
	const double zEstimate = Sampler.IntegratePathSampling(pIntegrandPS, pWidthPS, NULL) - std::log(2.0);
	// Estimate the weighting factor for the terminal distribution
	const double wEstimate = Sampler.Integrate(pIntegrandFS, NULL);

	std::cout << zEstimate << " " << std::log(wEstimate) << " " << zEstimate + std::log(wEstimate) << endl;
}

}  // namespace my_smctc
