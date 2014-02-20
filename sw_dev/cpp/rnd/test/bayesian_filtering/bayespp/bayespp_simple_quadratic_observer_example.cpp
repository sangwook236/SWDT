#include <BayesFilter/infFlt.hpp>
#include <Test/random.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/random.hpp>
#include <iostream>
#include <cmath>


namespace {
namespace local {

// square.
template <class scalar>
inline scalar sqr(scalar x)
{
	return x * x;
}

// random numbers from Boost.
Bayesian_filter_test::Boost_random localRng;

// constant dimensions.
const unsigned int NX = 3;  // filter state dimension, (system state, scale, bias).

// filter parameters.
// noise on observing system state.
const Bayesian_filter_matrix::Float OBS_NOISE = 0.01;
// prediction noise : no prediction noise as pertubation is known.
const Bayesian_filter_matrix::Float X_NOISE = 0.0;  // System State.
const Bayesian_filter_matrix::Float S_NOISE = 0.0;  // Scale.
const Bayesian_filter_matrix::Float B_NOISE = 0.0;  // Bias.
// filter's initial state uncertainty : system state is unknown.
const Bayesian_filter_matrix::Float i_X_NOISE = 1000.0;
const Bayesian_filter_matrix::Float i_S_NOISE = 0.1;
const Bayesian_filter_matrix::Float i_B_NOISE = 0.1;

// prediction model : linear state predict model with additive control input.
class QCpredict : public Bayesian_filter::Linrz_predict_model
{
public:
	QCpredict();

public:
	void predict(const Bayesian_filter_matrix::Vec &u)
	{
		motion = u[0];
	}
	const Bayesian_filter_matrix::Vec & f(const Bayesian_filter_matrix::Vec &x) const
	{
		// constant scale and bias, system state perturbed by control input.
		fx = x;
		fx[0] += motion;
		return fx;
	};

private:
	Bayesian_filter_matrix::Float motion;
	mutable Bayesian_filter_matrix::Vec fx;
};

QCpredict::QCpredict() : Bayesian_filter::Linrz_predict_model(NX, NX), fx(NX)
{
	Bayesian_filter_matrix::identity(Fx);

	// setup constant noise model : G is identity.
	q[0] = sqr(X_NOISE);
	q[1] = sqr(S_NOISE);
	q[2] = sqr(B_NOISE);

	Bayesian_filter_matrix::identity(G);
}


// quadratic observation model.
class QCobserve : public Bayesian_filter::Linrz_uncorrelated_observe_model
{
public:
	QCobserve();

public:
	// quadratic Observation model.
	const Bayesian_filter_matrix::Vec & h(const Bayesian_filter_matrix::Vec &x) const
	{
		z_pred[0] = x[0] * x[1] + x[2];
		return z_pred;
	};
	// linearised model, Jacobian of h at x.
	void state(const Bayesian_filter_matrix::Vec &x)
	{
		Hx(0, 0) = x[1];
		Hx(0, 1) = x[0];
		Hx(0, 2) = 1.;
	}

private:
	mutable Bayesian_filter_matrix::Vec z_pred;
};

QCobserve::QCobserve()
: Bayesian_filter::Linrz_uncorrelated_observe_model(NX, 1), z_pred(1)
{
	// observation noise variance.
	Zv[0] = OBS_NOISE * OBS_NOISE;
}

}  // namespace local
}  // unnamed namespace

namespace my_bayespp {

/*
 * Example of using Bayesian Filter Class to solve a simple problem.
 *
 * The example implements a simple quadratic observer.
 *  This tries to estimate the state of system while also trying to
 *  calibrate a simple linear model of the system which includes
 *  a scale factor and a bias.
 *  Estimating both the system state and a scale factor results in a
 *  quadratic (product of two states and therefore non-linear) observation.
 *  The system model is a 1D brownian motion with a known perturbation.
 */

// ${BAYES++_HOME}/QuadCalib/QuadCalib.cpp.
void simple_quadratic_observer_example()
{
	// global setup for test output.
	std::cout.flags(std::ios::scientific);
	std::cout.precision(6);

	// setup the test filters.
	Bayesian_filter_matrix::Vec x_true(local::NX);

	// true state to be observed.
	x_true[0] = 10.0;  // system state.
	x_true[1] = 1.0;  // scale.
	x_true[2] = 0.0;  // bias.

	std::cout << "Quadratic Calibration" << std::endl;
	std::cout << "Init : " << x_true << std::endl;

	// construct prediction and observation models and calibration filter.
	local::QCpredict linearPredict;
	local::QCobserve nonlinObserve;
	Bayesian_filter::Information_scheme obsAndCalib(local::NX);

	// give the filter an true initial guess of the system state.
	obsAndCalib.x[0] = x_true[0];
	obsAndCalib.x[1] = 1.0;  // assumed initial scale.
	obsAndCalib.x[2] = 0.0;  // assumed initial bias.
	obsAndCalib.X.clear();
	obsAndCalib.X(0, 0) = local::sqr(local::i_X_NOISE);
	obsAndCalib.X(1, 1) = local::sqr(local::i_S_NOISE);
	obsAndCalib.X(2, 2) = local::sqr(local::i_B_NOISE);

	obsAndCalib.init();

	// iterate the filter with test observations.
	Bayesian_filter_matrix::Vec u(1), z_true(1), z(1);
	for (unsigned i = 0; i < 100; i++ )
	{
		// predict true state using Brownian control input.
		local::localRng.normal(u);  // normally distributed.
		x_true[0] += u[0];
		linearPredict.predict(u);

		// predict filter with known perturbation.
		obsAndCalib.predict(linearPredict);

		// true observation : quadratic observation model.
		z_true[0] = x_true[0] * x_true[1] + x_true[2];

		// observation with additive noise.
		local::localRng.normal(z, z_true[0], local::OBS_NOISE);  // normally distributed mean z_true[0], stdDev OBS_NOISE.

		// filter observation using model linearised at state estimate x.
		nonlinObserve.state(obsAndCalib.x);
		obsAndCalib.observe(nonlinObserve, z);
	}

	// update the filter to state and covariance are available.
	obsAndCalib.update();

	// print everything : true, filter, covariance.
	std::cout << "True : " << x_true <<  std::endl;
	std::cout << "Calb : " << obsAndCalib.x << std::endl;
	std::cout << obsAndCalib.X << std::endl;
}

}  // namespace my_bayespp
