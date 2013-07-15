#include "pffuncs.h"
#include <smctc/smctc.hh>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>


namespace {
namespace local {

long load_data(char const *szName, pffuncs::cv_obs **yp)
{
	FILE *fObs = fopen(szName, "rt");
	if (!fObs)
		throw SMC_EXCEPTION(SMCX_FILE_NOT_FOUND, "Error: pf assumes that the current directory contains an appropriate data file called data.csv\nThe first line should contain a constant indicating the number of data lines it contains.\nThe remaining lines should contain comma-separated pairs of x, y observations.");

	char *szBuffer = new char[1024];
	fgets(szBuffer, 1024, fObs);
	long lIterates = strtol(szBuffer, NULL, 10);

	*yp = new pffuncs::cv_obs [lIterates];

	for (long i = 0; i < lIterates; ++i)
	{
		fgets(szBuffer, 1024, fObs);
		(*yp)[i].x_pos = strtod(strtok(szBuffer, ",\r\n "), NULL);
		(*yp)[i].y_pos = strtod(strtok(NULL, ",\r\n "), NULL);
	}
	fclose(fObs);

	delete [] szBuffer;

	return lIterates;
}

double integrand_mean_x(const pffuncs::cv_state &s, void *)
{
	return s.x_pos;
}

double integrand_var_x(const pffuncs::cv_state &s, void *vmx)
{
	double *dmx = (double *)vmx;
	double d = (s.x_pos - (*dmx));
	return d * d;
}

double integrand_mean_y(const pffuncs::cv_state &s, void *)
{
	return s.y_pos;
}

double integrand_var_y(const pffuncs::cv_state &s, void *vmy)
{
	double *dmy = (double *)vmy;
	double d = (s.y_pos - (*dmy));
	return d * d;
}

}  // namespace local
}  // unnamed namespace

namespace my_smctc {

// [ref] ${SMCTC_HOME}/examples/pf/pfexample.cc
void pf_example()
{
	// Number of Particles
	const long lNumber = 1000;

	// Load observations
	const long lIterates = local::load_data("./bayesian_filtering/data.csv", &pffuncs::y);

	// Initialise and run the sampler
	smc::sampler<pffuncs::cv_state> Sampler(lNumber, SMC_HISTORY_NONE);
	smc::moveset<pffuncs::cv_state> Moveset(pffuncs::fInitialise, pffuncs::fMove, NULL);

	Sampler.SetResampleParams(SMC_RESAMPLE_RESIDUAL, 0.5);
	Sampler.SetMoveSet(Moveset);
	Sampler.Initialise();

	for (int n = 1; n < lIterates; ++n)
	{
		Sampler.Iterate();

		double xm, xv, ym, yv;
		xm = Sampler.Integrate(local::integrand_mean_x, NULL);
		xv = Sampler.Integrate(local::integrand_var_x, (void *)&xm);
		ym = Sampler.Integrate(local::integrand_mean_y, NULL);
		yv = Sampler.Integrate(local::integrand_var_y, (void *)&ym);

		std::cout << xm << ", " << ym << ", " << xv << ", " << yv << std::endl;
	}
}

}  // namespace my_smctc
