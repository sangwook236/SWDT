#include <libcmaes/cmaes.h>
#include <iostream>
#include <vector>


namespace {
namespace local {

libcmaes::FitFunc fsphere = [](const double *x, const int N)
{
	double val = 0.0;
	for (int i = 0; i < N; ++i)
		val += x[i] * x[i];
	return val;
};

libcmaes::GradFunc grad_fsphere = [](const double *x, const int N)
{
	dVec grad(N);
	for (int i = 0; i < N; ++i)
		grad(i) = 2.0 * x[i];
	return grad;
};

// REF [file] >> ${LIBCMAES_HOME}/examples/sample-code.cc
int sample_code()
{
	const int dim = 10;  // Problem dimensions.
	const std::vector<double> x0(dim, 10.0);
	const double sigma = 0.1;
	//const int lambda = 100;  // Offsprings at each generation.

	libcmaes::CMAParameters<> cmaparams(x0, sigma);
	//cmaparams._algo = BIPOP_CMAES;
	const libcmaes::CMASolutions &cmasols = libcmaes::cmaes<>(fsphere, cmaparams);

	std::cout << "Best solution: " << cmasols << std::endl;
	std::cout << "Optimization took " << cmasols.elapsed_time() / 1000.0 << " seconds." << std::endl;
	return cmasols.run_status();
}

// REF [file] >> ${LIBCMAES_HOME}/examples/sample-code-bounds.cc
int sample_code_bounds()
{
	const int dim = 10;  // Problem dimensions.
	const double sigma = 0.1;
	double lbounds[dim], ubounds[dim];  // Arrays for lower and upper parameter bounds, respectively.
	for (int i = 0; i < dim; ++i)
	{
		lbounds[i] = -2.0;
		ubounds[i] = 2.0;
	}
	const std::vector<double> x0(dim, 1.0);  // Beware that x0 is within bounds.

	libcmaes::GenoPheno<libcmaes::pwqBoundStrategy> gp(lbounds, ubounds, dim);  // Genotype/phenotype transform associated to bounds.
	libcmaes::CMAParameters<libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>> cmaparams(x0, sigma, -1, 0, gp);  // -1 for automatically decided lambda, 0 is for random seeding of the internal generator.
	cmaparams.set_algo(aCMAES);
	const libcmaes::CMASolutions &cmasols = libcmaes::cmaes<libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>>(fsphere, cmaparams);

	std::cout << "Best solution: ";
	cmasols.print(std::cout, 0, gp);
	std::cout << std::endl;
	std::cout << "Optimization took " << cmasols.elapsed_time() / 1000.0 << " seconds." << std::endl;
	return cmasols.run_status();
}

// REF [file] >> ${LIBCMAES_HOME}/examples/sample-code-gradient.cc
int sample_code_gradient()
{
	const int dim = 10;  // Problem dimensions.
	const std::vector<double> x0(dim, 10.0);
	const double sigma = 0.1;
	//const int lambda = 100;  // Offsprings at each generation.

	libcmaes::CMAParameters<> cmaparams(x0, sigma);
	cmaparams.set_algo(aCMAES);
	const libcmaes::CMASolutions &cmasols = libcmaes::cmaes<>(fsphere, cmaparams,
		libcmaes::CMAStrategy<libcmaes::CovarianceUpdate>::_defaultPFunc,  // Use default progress function.
		grad_fsphere);

	std::cout << "Best solution: " << cmasols << std::endl;
	std::cout << "Optimization took " << cmasols.elapsed_time() / 1000.0 << " seconds." << std::endl;
	return cmasols.run_status();
}

// Dummy genotype/phenotype transform functions.
libcmaes::TransFunc genof = [](const double *ext, double *in, const int &dim)
{
	for (int i = 0; i < dim; ++i)
		in[i] = 2.0 * ext[i];
};

libcmaes::TransFunc phenof = [](const double *in, double *ext, const int &dim)
{
	for (int i = 0; i < dim; ++i)
		ext[i] = 0.5 * in[i];
};

// REF [file] >> ${LIBCMAES_HOME}/examples/sample-code-genopheno.cc
int sample_code_genopheno()
{
	const int dim = 10;  // Problem dimensions.
	const std::vector<double> x0(dim, 1.0);
	const double sigma = 0.1;
	//const int lambda = 100;  // Offsprings at each generation.

	libcmaes::GenoPheno<> gp(genof, phenof);
	libcmaes::CMAParameters<> cmaparams(x0, sigma, -1, 0, gp);  // -1 for automatically decided lambda.
	//cmaparams._algo = BIPOP_CMAES;
	const libcmaes::CMASolutions &cmasols = libcmaes::cmaes<>(fsphere, cmaparams);

	std::cout << "Best solution: " << cmasols << std::endl;
	std::cout << "Optimization took " << cmasols.elapsed_time() / 1000.0 << " seconds." << std::endl;
	return cmasols.run_status();
}

// REF [file] >> ${LIBCMAES_HOME}/examples/sample-code-lscaling.cc
int sample_code_lscaling()
{
	const int dim = 10;  // Problem dimensions.
	const std::vector<double> x0(dim, 1.0);
	const double sigma = 0.1;
	double lbounds[dim], ubounds[dim];  // Arrays for lower and upper parameter bounds, respectively.
	for (int i = 0; i<dim; i++)
	{
		lbounds[i] = -2.0;
		ubounds[i] = 2.0;
	}

	libcmaes::GenoPheno<libcmaes::pwqBoundStrategy, libcmaes::linScalingStrategy> gp(lbounds, ubounds, dim);
	libcmaes::CMAParameters<libcmaes::GenoPheno<libcmaes::pwqBoundStrategy, libcmaes::linScalingStrategy>> cmaparams(x0, sigma, -1, 0, gp);  // -1 for automatically decided lambda.
	cmaparams.set_algo(aCMAES);
	const libcmaes::CMASolutions &cmasols = libcmaes::cmaes<libcmaes::GenoPheno<libcmaes::pwqBoundStrategy, libcmaes::linScalingStrategy>>(fsphere, cmaparams);

	std::cout << "Best solution: ";
	cmasols.print(std::cout, 0, gp);
	std::cout << std::endl;
	std::cout << "Optimization took " << cmasols.elapsed_time() / 1000.0 << " seconds." << std::endl;
	return cmasols.run_status();
}

// REF [file] >> ${LIBCMAES_HOME}/examples/sample-code-lscaling-sigmas.cc
int sample_code_lscaling_sigmas()
{
	const std::vector<double> x0 = { 1.0, 2.7, 400.0 };
	const std::vector<double> sigmas = { 1e-3, 0.57, 2.3 };

	libcmaes::CMAParameters<libcmaes::GenoPheno<libcmaes::NoBoundStrategy, libcmaes::linScalingStrategy>> cmaparams(x0, sigmas);
	cmaparams.set_algo(aCMAES);
	const libcmaes::CMASolutions &cmasols = libcmaes::cmaes<libcmaes::GenoPheno<libcmaes::NoBoundStrategy, libcmaes::linScalingStrategy>>(fsphere, cmaparams);

	std::cout << "Best solution: " << cmasols << std::endl;
	std::cout << "Optimization took " << cmasols.elapsed_time() / 1000.0 << " seconds." << std::endl;
	return cmasols.run_status();
}

libcmaes::FitFunc rosenbrock = [](const double *x, const int N)
{
	double val = 0.0;
	for (int i = 0; i < N - 1; ++i)
	{
		val += 100.0 * std::pow((x[i + 1] - x[i] * x[i]), 2) + std::pow((x[i] - 1.0), 2);
	}
	return val;
};

libcmaes::PlotFunc<libcmaes::CMAParameters<>, libcmaes::CMASolutions> plotf = [](const libcmaes::CMAParameters<> &cmaparams, const libcmaes::CMASolutions &cmasols, std::ofstream &fplotstream)
{
	fplotstream << "kappa=" << cmasols.max_eigenv() / cmasols.min_eigenv() << std::endl;  // Storing covariance matrix condition number to file.
	return 0;
};

// REF [file] >> ${LIBCMAES_HOME}/examples/sample-code-pffunc.cc
int sample_code_pffunc()
{
	const int dim = 20;  // Problem dimensions.
	const std::vector<double> x0(dim, 10.0);
	const double sigma = 0.1;
	//const int lambda = 100;  // Ofsprings at each generation.

	libcmaes::CMAParameters<> cmaparams(x0, sigma);
	cmaparams.set_fplot("pffunc.dat");  // DON'T MISS: mandatory output file name.
	const libcmaes::CMASolutions &cmasols = libcmaes::cmaes<>(rosenbrock, cmaparams, libcmaes::CMAStrategy<libcmaes::CovarianceUpdate>::_defaultPFunc, nullptr, libcmaes::CMASolutions(), plotf);

	std::cout << "Best solution: " << cmasols << std::endl;
	std::cout << "Optimization took " << cmasols.elapsed_time() / 1000.0 << " seconds." << std::endl;
	return cmasols.run_status();
}

libcmaes::ProgressFunc<libcmaes::CMAParameters<>, libcmaes::CMASolutions> select_time = [](const libcmaes::CMAParameters<> &cmaparams, const libcmaes::CMASolutions &cmasols)
{
	if (cmasols.niter() % 100 == 0)
		std::cerr << cmasols.elapsed_last_iter() << std::endl;
	return 0;
};

// REF [file] >> ${LIBCMAES_HOME}/examples/sample-code-pfunc.cc
int sample_code_pfunc()
{
	const int dim = 100;  // Problem dimensions.
	const std::vector<double> x0(dim, 10.0);
	const double sigma = 0.1;
	//const int lambda = 100;  // Offsprings at each generation.

	libcmaes::CMAParameters<> cmaparams(x0, sigma);
	//cmaparams._algo = BIPOP_CMAES;
	const libcmaes::CMASolutions &cmasols = libcmaes::cmaes<>(rosenbrock, cmaparams, select_time);

	std::cout << "Best solution: " << cmasols << std::endl;
	std::cout << "Optimization took " << cmasols.elapsed_time() / 1000.0 << " seconds." << std::endl;
	return cmasols.run_status();
}

}  // namespace local
}  // unnamed namespace

namespace my_libcmaes {

}  // namespace my_libcmaes

int libcmaes_main(int argc, char *argv[])
{
	const int status1 = local::sample_code();
	const int status2 = local::sample_code_bounds();
	const int status3 = local::sample_code_gradient();
	const int status4 = local::sample_code_genopheno();
	// Linear scaling.
	const int status5 = local::sample_code_lscaling();
	const int status6 = local::sample_code_lscaling_sigmas();

	// Plot.
	const int status7 = local::sample_code_pffunc();
	// Progress.
	//const int status8 = local::sample_code_pfunc();

	return 0;
}

