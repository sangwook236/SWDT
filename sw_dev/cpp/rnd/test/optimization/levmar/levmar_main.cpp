//#include "stdafx.h"
#include <levmar/lm.h>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <vector>
#include <iostream>
#include <ctime>
#include <cmath>


#if defined(max)
#undef max
#endif

namespace {
namespace local {

// Meyer's (reformulated) problem, minimum at (2.48, 6.18, 3.45)
static void meyer(double* params, double* func, int dimParams, int dimFuncs, void* data)
{
	for (register int i = 0; i < dimFuncs; ++i)
	{
		const double ui = 0.45 + 0.05 * i;
		func[i] = params[0] * std::exp(10.0 * params[1] / (ui + params[2]) - 13.0);
	}
}

void jacmeyer(double* params, double* jacobian, int dimParams, int dimFuncs, void* data)
{
	for (register int i = 0, j = 0; i < dimFuncs; ++i)
	{
		const double ui = 0.45 + 0.05*i;
		const double tmp = exp(10.0*params[1] / (ui + params[2]) - 13.0);

		// jacobian: dimMeasures-by-dimParams
		jacobian[j++] = tmp;
		jacobian[j++] = 10.0*params[0]*tmp / (ui + params[2]);
		jacobian[j++] = -10.0*params[0]*params[1]*tmp / ((ui + params[2]) * (ui + params[2]));
	}
}

bool levenberg_marquardt__nonlinear_least_square__unconstrained_optimization__1()
{
    const int dimParams = 3;  // parameter vector dimension (i.e. #unknowns)
	const int dimMeasures = 16;  // measurement vector dimension
	assert(dimMeasures >= dimParams);

	double params[dimParams];  // initial parameter estimates. On output contains the estimated solution
    params[0] = 8.85;
	params[1] = 4.0;
	params[2] = 2.5;
	double measures[dimMeasures];  // measurement vector: not changed -> given function values
	measures[0] = 34.780;
	measures[1] = 28.610;
	measures[2] = 23.650;
	measures[3] = 19.630;
    measures[4] = 16.370;
	measures[5] = 13.720;
	measures[6] = 11.540;
	measures[7] = 9.744;
    measures[8] = 8.261;
	measures[9] = 7.030;
	measures[10] = 6.005;
	measures[11] = 5.147;
    measures[12] = 4.427;
	measures[13] = 3.820;
	measures[14] = 3.307;
	measures[15] = 2.872;

	//
	double* work = new double [LM_DIF_WORKSZ(dimParams, dimMeasures) + dimParams*dimParams];
	//double* work = new double [LM_DER_WORKSZ(dimParams, dimMeasures) + dimParams*dimParams];
	if (!work)
	{
		std::cout << "memory allocation request failed" << std::endl;
		return false;
	}
	double* covar = work + LM_DIF_WORKSZ(dimParams, dimMeasures);
	//double* covar = work + LM_DER_WORKSZ(dimParams, dimMeasures);

	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0] = LM_INIT_MU;
	opts[1] = 1.0e-15;
	opts[2] = 1.0e-15;
	opts[3] = 1.0e-20;
	opts[4] = LM_DIFF_DELTA;  // relevant only if the finite difference jacobian version is used 

	const int maxIteration = 1000;
	const int ret = dlevmar_dif(meyer, params, measures, dimParams, dimMeasures, maxIteration, opts, info, work, covar, NULL); // no jacobian, caller allocates work memory, covariance estimated
    //const int ret = dlevmar_der(meyer, jacmeyer, params, measures, dimParams, dimMeasures, maxIteration, opts, info, work, covar, NULL); // with analytic jacobian
	assert(ret >= 0);

	//
	std::cout << "solution:" << std::endl;
	for (int i = 0; i < dimParams; ++i)
		std::cout << params[i] << ' ';
	std::cout << std::endl;

	std::cout << "covariance of the fit:" << std::endl;
	for (int i = 0; i < dimParams; ++i)
	{
		for (int j = 0; j < dimParams; ++j)
			std::cout << covar[i*dimParams+j] << ' ';
		std::cout << std::endl;
	}
	std::cout << std::endl;

	delete [] work;
	return true;
}

static void objective_f(double* params, double* func, int dimParams, int dimFuncs, void* data)
{
	const double a0 = params[0];
	const double a1 = params[1];
	const double a2 = params[2];
	const double a3 = params[3];
	const double a4 = params[4];
	const double a5 = params[5];

	for (register int i = 0; i < dimFuncs; ++i)
	{
		// unknowns: a0, a1, a2, a3, a4, a5
		// model: fi = a0 + a1 * i + a2 * i^2 + a3 * i^3 + a4 * i^4 + a5 * i^5
		const double t = i;
		func[i] = a0 + a1 * t + a2 * t*t + a3 * t*t*t + a4 * t*t*t*t + a5 * t*t*t*t*t;
	}
}

static void objective_df(double* params, double* jacobian, int dimParams, int dimFuncs, void* data)
{
/*
	const double a0 = params[0];
	const double a1 = params[1];
	const double a2 = params[2];
	const double a3 = params[3];
	const double a4 = params[4];
	const double a5 = params[5];
*/
	for (register int i = 0, j = 0; i < dimFuncs; ++i)
	{
		// unknowns: a0, a1, a2, a3, a4, a5
		// Jacobian matrix J(i,j) = dfi / dxj
		//   where fi = a0 + a1 * i + a2 * i^2 + a3 * i^3 + a4 * i^4 + a5 * i^5
		//         xj = the parameters (a0, a1, a2, a3, a4, a5)

		// jacobian: dimMeasures-by-dimParams
		const double t = i;
		jacobian[j++] = 1.0;
		jacobian[j++] = t;
		jacobian[j++] = t*t;
		jacobian[j++] = t*t*t;
		jacobian[j++] = t*t*t*t;
		jacobian[j++] = t*t*t*t*t;
	}
}

bool levenberg_marquardt__nonlinear_least_square__unconstrained_optimization__2()
{
	srand((unsigned int)time(NULL));
	//const int measureCount = rand();
	const int measureCount = 40;

    const int dimParams = 6;  // parameter vector dimension (i.e. #unknowns)
	const int dimMeasures = std::max(measureCount, dimParams);  // measurement vector dimension
	assert(measureCount >= dimParams);

	double params[dimParams] = { 0.0, };  // initial parameter estimates. On output contains the estimated solution
	std::vector<double> measures(dimMeasures, 0.0);  // measurement vector: not changed -> given function values
	{
/*
		typedef boost::minstd_rand base_generator_type;
		typedef boost::normal_distribution<> distribution_type;
		typedef boost::variate_generator<base_generator_type&, distribution_type> generator_type;

		base_generator_type generator(42u);
		const double mean = 0.0;
		const double sigma = 1.0;
		generator_type white_noise_gen(generator, distribution_type(mean, sigma));

		for (int i = 0; i < measureCount; ++i)
		{
			const double t = i;
			measures[i] = 2.0 + 4.5 * t - 2.3 * t*t - 11.7 * t*t*t + 0.3 * t*t*t*t - 8.4 * t*t*t*t*t + white_noise_gen();
		};
*/
		//
		typedef boost::minstd_rand base_generator_type;
		typedef boost::variate_generator<base_generator_type &, boost::normal_distribution<> > generator_type;

		base_generator_type baseGenerator(static_cast<unsigned int>(std::time(NULL)));
		generator_type generator(baseGenerator, boost::normal_distribution<>(0.0, 1.0));

		for (size_t i = 0; i < measureCount; ++i)
		{
			const double t = i;
			measures[i] = 2.0 + 4.5 * t - 2.3 * t*t - 11.7 * t*t*t + 0.3 * t*t*t*t - 8.4 * t*t*t*t*t + generator();
		};
	}

	//
	//double* work = new double [LM_DIF_WORKSZ(dimParams, dimMeasures) + dimParams*dimParams];
	double* work = new double [LM_DER_WORKSZ(dimParams, dimMeasures) + dimParams*dimParams];
	if (!work)
	{
		std::cout << "memory allocation request failed" << std::endl;
		return false;
	}
	//double* covar = work + LM_DIF_WORKSZ(dimParams, dimMeasures);
	double* covar = work + LM_DER_WORKSZ(dimParams, dimMeasures);

	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0] = LM_INIT_MU;
	opts[1] = 1.0e-15;
	opts[2] = 1.0e-15;
	opts[3] = 1.0e-20;
	opts[4] = LM_DIFF_DELTA;  // relevant only if the finite difference jacobian version is used 

	const int maxIteration = 1000;
	//const int ret = dlevmar_dif(objective_f, params, &measures[0], dimParams, dimMeasures, maxIteration, opts, info, work, covar, NULL); // no jacobian, caller allocates work memory, covariance estimated
	const int ret = dlevmar_der(objective_f, objective_df, params, &measures[0], dimParams, dimMeasures, maxIteration, opts, info, work, covar, NULL); // with analytic jacobian
	assert(ret >= 0);

	//
	{
		std::vector<double> resultMeasures(dimMeasures, 0.0);;
		objective_f(params, &resultMeasures[0], dimParams, dimMeasures, NULL);

		std::cout << "covariance of the fit:" << std::endl;
		for (int i = 0; i < dimParams; ++i)
		{
			for (int j = 0; j < dimParams; ++j)
				std::cout << covar[i*dimParams+j] << ' ';
			std::cout << std::endl;
		}

		double chi = 0.0;
		for (int i = 0; i < measureCount; ++i)
			chi += std::pow(resultMeasures[i] - measures[i], 2.0);
		chi = std::sqrt(chi);
		const double dof = measureCount - dimParams;
		const double c = std::max(1.0, chi / std::sqrt(dof));
		std::cout << "chisq/dof = " << std::pow(chi, 2.0) / dof << std::endl;

		std::cout << "solution:" << std::endl;
		std::cout << "a0 = " << params[0] << " +/- " << c * std::sqrt(covar[0*dimParams+0]) << std::endl;
		std::cout << "a1 = " << params[1] << " +/- " << c * std::sqrt(covar[1*dimParams+1]) << std::endl;
		std::cout << "a2 = " << params[2] << " +/- " << c * std::sqrt(covar[2*dimParams+2]) << std::endl;
		std::cout << "a3 = " << params[3] << " +/- " << c * std::sqrt(covar[3*dimParams+3]) << std::endl;
		std::cout << "a4 = " << params[4] << " +/- " << c * std::sqrt(covar[4*dimParams+4]) << std::endl;
		std::cout << "a5 = " << params[5] << " +/- " << c * std::sqrt(covar[5*dimParams+5]) << std::endl;
	}

	delete [] work;
	return true;
}

}  // namespace local
}  // unnamed namespace

namespace levmar {

}  // namespace levmar

int levmar_main(int argc, char *argv[])
{
	local::levenberg_marquardt__nonlinear_least_square__unconstrained_optimization__1();
	local::levenberg_marquardt__nonlinear_least_square__unconstrained_optimization__2();

    return 0;
}
