#include <BayesFilter/UDFlt.hpp>
#include <BayesFilter/filters/indirect.hpp>
#include <Test/random.hpp>
#include <boost/numeric/ublas/io.hpp>
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
const unsigned int NX = 2;  // filter state dimension, (position, velocity).

// filter parameters.
// prediction parameters for integrated Ornstein-Uhlembeck process.
const Bayesian_filter_matrix::Float dt = 0.01;
const Bayesian_filter_matrix::Float V_NOISE = 0.1;  // velocity noise, giving mean squared error bound.
const Bayesian_filter_matrix::Float V_GAMMA = 1.0;  // velocity correlation, giving velocity change time constant.
// filter's initial state uncertainty : system state is unknown.
const Bayesian_filter_matrix::Float i_P_NOISE = 1000.0;
const Bayesian_filter_matrix::Float i_V_NOISE = 10.0;
// noise on observing system state.
const Bayesian_filter_matrix::Float OBS_INTERVAL = 0.1;
const Bayesian_filter_matrix::Float OBS_NOISE = 0.001;

// prediction model : linear state predict model.
class PVpredict : public Bayesian_filter::Linear_predict_model
{
public:
	PVpredict();
};

PVpredict::PVpredict() : Bayesian_filter::Linear_predict_model(NX, 1)
{
	// position-velocity dependence.
	const Float Fvv = std::exp(-dt * V_GAMMA);
	Fx(0, 0) = 1.0;
	Fx(0, 1) = dt;
	Fx(1, 0) = 0.0;
	Fx(1, 1) = Fvv;

	// setup constant noise model : G is identity.
	q[0] = dt * sqr((1 - Fvv) * V_NOISE);
	G(0, 0) = 0.0;
	G(1, 0) = 1.0;
}

// position observation model : linear observation is additive uncorrelated model.
class PVobserve : public Bayesian_filter::Linrz_uncorrelated_observe_model
{
public:
	PVobserve();

public:
	const Bayesian_filter_matrix::Vec & h(const Bayesian_filter_matrix::Vec &x) const
	{
		z_pred[0] = x[0];
		return z_pred;
	};

private:
	mutable Bayesian_filter_matrix::Vec z_pred;
};

PVobserve::PVobserve()
:Linrz_uncorrelated_observe_model(NX, 1), z_pred(1)
{
	// linear model.
	Hx(0, 0) = 1.0;
	Hx(0, 1) = 0.0;

	// observation noise variance.
	Zv[0] = sqr(OBS_NOISE);
}

// initialise Kalman filter with an initial guess for the system state and fixed covariance.
void initialise(Bayesian_filter::Kalman_state_filter &kf, const Bayesian_filter_matrix::Vec &initState)
{
	// initialise state guess and covarince.
	kf.X.clear();
	kf.X(0, 0) = sqr(i_P_NOISE);
	kf.X(1, 1) = sqr(i_V_NOISE);

	kf.init_kalman(initState, kf.X);
}

}  // namespace local
}  // unnamed namespace

namespace my_bayespp {

/*
 * Example of using Bayesian Filter Class to solve a simple problem.
 *  The example implements a Position and Velocity Filter with a Position observation.
 *  The motion model is the so called IOU Integrated Ornstein-Uhlenbeck Process Ref[1]
 *    Velocity is Brownian with a trend towards zero proportional to the velocity
 *    Position is just Velocity integrated.
 *  This model has a well defined velocity and the mean squared speed is parameterised. Also
 *  the velocity correlation is parameterised.
 *  
 * Two implementations are demonstrated
 *  1) A direct filter
 *  2) An indirect filter where the filter is performed on error and state is estimated indirectly
 *
 * Reference
 *	[1] "Bayesian Multiple Target Tracking" Lawrence D Stone, Carl A Barlow, Thomas L Corwin
 */

// ${BAYES++_HOME}/PV/PV.cpp.
void position_and_velocity_filter_example()
{
	// global setup.
	std::cout.flags(std::ios::scientific);
	std::cout.precision(6);

	// setup the test filters.
	Bayesian_filter_matrix::Vec x_true(local::NX);

	// true state to be observed.
	x_true[0] = 1000.0;	 // position.
	x_true[1] = 1.0;  // velocity.

	std::cout << "Position-Velocity" << std::endl;
	std::cout << "True Initial : " << x_true << std::endl;

	// construct prediction and observation model and filter.
	// give the filter an initial guess of the system state.
	local::PVpredict linearPredict;
	local::PVobserve linearObserve;

	Bayesian_filter_matrix::Vec x_guess(local::NX);
	x_guess[0] = 900.0;
	x_guess[1] = 1.5;
	std::cout << "Guess Initial : " << x_guess << std::endl;

	// f1 direct filter construct and initialize with initial state guess.
	Bayesian_filter::UD_scheme f1(local::NX, local::NX);
	local::initialise(f1, x_guess);

	// f2 indirect filter construct and Initialize with initial state guess.
	Bayesian_filter::UD_scheme error_filter(local::NX, local::NX);
	Bayesian_filter::Indirect_kalman_filter<Bayesian_filter::UD_scheme> f2(error_filter);
	local::initialise(f2, x_guess);

	// iterate the filter with test observations.
	Bayesian_filter_matrix::Vec u(1), z_true(1), z(1);
	Bayesian_filter_matrix::Float time = 0.0;
	Bayesian_filter_matrix::Float obs_time = 0.0;
	for (unsigned int i = 0; i < 100; ++i)
	{
		// predict true state using normally distributed acceleration.
		// this is a Guassian.
		x_true = linearPredict.f(x_true);
		local::localRng.normal(u);  // normally distributed mean 0., stdDev for stationary IOU.
		x_true[1] += u[0] * local::sqr(local::V_NOISE) / (2 * local::V_GAMMA);

		// predict filter with known perturbation.
		f1.predict(linearPredict);
		f2.predict(linearPredict);
		time += local::dt;

		// observation time.
		if (obs_time <= time)
		{
			// true Observation.
			z_true[0] = x_true[0];

			// observation with additive noise.
			local::localRng.normal(z, z_true[0], local::OBS_NOISE);  // normally distributed mean z_true[0], stdDev OBS_NOISE.

			// filter observation.
			f1.observe(linearObserve, z);
			f2.observe(linearObserve, z);

			obs_time += local::OBS_INTERVAL;
		}
	}

	// update the filter to state and covariance are available.
	f1.update();
	f2.update();

	// print everything: filter state and covariance.
	std::cout << "True     : " << x_true << std::endl;
	std::cout << "Direct   : " << f1.x << ',' << f1.X << std::endl;
	std::cout << "Indirect : " << f2.x << ',' << f2.X << std::endl;
}

}  // namespace my_bayespp
