#include <BayesFilter/unsFlt.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>


namespace {
namespace local {

// simple Prediction model.
class Simple_predict : public Bayesian_filter::Linear_predict_model
{
public:
	// construct a constant model.
	Simple_predict() : Bayesian_filter::Linear_predict_model(1, 1)
	{
		// stationary prediction model (Identity).
		Fx(0, 0) = 1.0;

		// constant noise model with a large variance.
		q[0] = 2.0;
		G(0, 0) = 1.0;
	}
};

// simple Observation model.
class Simple_observe : public Bayesian_filter::Linear_uncorrelated_observe_model
{
public:
	// construct a constant model.
	Simple_observe() : Bayesian_filter::Linear_uncorrelated_observe_model(1, 1)
	{
		// linear model.
		Hx(0, 0) = 1.0;

		// constant observation noise model with variance of one.
		Zv[0] = 1.0;
	}
};

}  // namespace local
}  // unnamed namespace

namespace my_bayespp {

// ${BAYES++_HOME}/Simple/simpleExample.cpp.
void simple_example()
{
	// global setup for test output.
	std::cout.flags(std::ios::fixed);
	std::cout.precision(4);

	// construct simple Prediction and Observation models.
	local::Simple_predict my_predict;
	local::Simple_observe my_observe;

	// use an 'Unscented' filter scheme with one state.
	Bayesian_filter::Unscented_scheme my_filter(1);

	// setup the initial state and covariance.
	Bayesian_filter_matrix::Vec x_init(1);
	Bayesian_filter_matrix::SymMatrix X_init(1, 1);
	x_init[0] = 10.0;  // start at 10 with no uncertainty.
	X_init(0, 0) = 0.0;

	// initialize from a state and state covariance.
	my_filter.init_kalman(x_init, X_init);

	std::cout << "Initial  " << my_filter.x << my_filter.X << std::endl;

	// predict the filter forward.
	my_filter.predict(my_predict);
	// update the filter, so state and covariance are available.
	my_filter.update();

	std::cout << "Predict  " << my_filter.x << my_filter.X << std::endl;

	// make an observation.
	Bayesian_filter_matrix::Vec z(1);
	z[0] = 11.0;  // observe that we should be at 11.
	my_filter.observe(my_observe, z);

	// update the filter to state and covariance are available.
	my_filter.update();

	std::cout << "Filtered " << my_filter.x << my_filter.X << std::endl;
}

}  // namespace my_bayespp
