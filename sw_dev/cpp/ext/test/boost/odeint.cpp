#include <boost/numeric/odeint.hpp>
#include <iostream>
#include <vector>


namespace {
namespace local {

typedef std::vector<double> state_type;

// REF [file] >> ${BOOST_HOME}/libs/numeric/odeint/examples/harmonic_oscillator.cpp
struct harmonic_oscillator
{
public:
	harmonic_oscillator(const double gamma)
	: gamma_(gamma)
	{}

	void operator()(const state_type &x, state_type &dxdt, const double /*t*/)
	{
		dxdt[0] = x[1];
		dxdt[1] = -x[0] - gamma_ * x[1];
	}

private:
	const double gamma_;
};

// REF [file] >> ${BOOST_HOME}/libs/numeric/odeint/examples/harmonic_oscillator.cpp
// Integration observer.
struct push_back_state_and_time
{
	push_back_state_and_time(std::vector<state_type> &states, std::vector<double> &times)
	: states_(states), times_(times)
	{}

	void operator()(const state_type &x, double t)
	{
		states_.push_back(x);
		times_.push_back(t);
	}

	std::vector<state_type>& states_;
	std::vector<double>& times_;
};

// REF [file] >> ${BOOST_HOME}/libs/numeric/odeint/examples/harmonic_oscillator.cpp
void harmonic_oscillator_example()
{
	const double gamma = 0.15;

	{
		state_type x(2);
		x[0] = 1.0;
		x[1] = 0.0;

		const size_t steps = boost::numeric::odeint::integrate(harmonic_oscillator(gamma), x, 0.0, 10.0, 0.01);
		std::cout << "x = (" << x[0] << ',' << x[1] << "), step = " << steps << std::endl;
	}

	// Use observer.
	{
		state_type x(2);
		x[0] = 1.0;
		x[1] = 0.0;

		std::vector<state_type> states;
		std::vector<double> times;

		const size_t steps = boost::numeric::odeint::integrate(harmonic_oscillator(gamma), x, 0.0, 10.0, 0.01, push_back_state_and_time(states, times));
		std::cout << "step = " << steps << std::endl;
		for (size_t i = 0; i <= steps; ++i)
			std::cout << "\tt = " << times[i] << ", x = (" << states[i][0] << ',' << states[i][1] << ')' << std::endl;
	}

	// Use stepper. Integrate with constant step size.
	{
		boost::numeric::odeint::runge_kutta4<state_type> stepper;

		{
			state_type x(2);
			x[0] = 1.0;
			x[1] = 0.0;

			const size_t steps = boost::numeric::odeint::integrate_const(stepper, harmonic_oscillator(gamma), x, 0.0, 10.0, 0.01);
			std::cout << "x = (" << x[0] << ',' << x[1] << "), step = " << steps << std::endl;
		}

		//
		{
			state_type x(2);
			x[0] = 1.0;
			x[1] = 0.0;

			const double dt = 0.01;
			for (double t = 0.0; t < 10.0; t += dt)
				stepper.do_step(harmonic_oscillator(0.15), x, t, dt);
			std::cout << "x = (" << x[0] << ',' << x[1] << ')' << std::endl;
		}
	}

	// Use adaptor. Integrate with adaptive step size.
	{
		typedef boost::numeric::odeint::runge_kutta_cash_karp54<state_type> error_stepper_type;
		typedef boost::numeric::odeint::controlled_runge_kutta<error_stepper_type> controlled_stepper_type;

		{
			state_type x(2);
			x[0] = 1.0;
			x[1] = 0.0;

			controlled_stepper_type controlled_stepper;
			const size_t steps = boost::numeric::odeint::integrate_adaptive(controlled_stepper, harmonic_oscillator(gamma), x, 0.0, 10.0, 0.01);
			std::cout << "x = (" << x[0] << ',' << x[1] << "), step = " << steps << std::endl;
		}

		{
			state_type x(2);
			x[0] = 1.0;
			x[1] = 0.0;

			const double abs_err = 1.0e-10, rel_err = 1.0e-6, a_x = 1.0, a_dxdt = 1.0;
			controlled_stepper_type controlled_stepper(boost::numeric::odeint::default_error_checker<double, boost::numeric::odeint::range_algebra, boost::numeric::odeint::default_operations>(abs_err, rel_err, a_x, a_dxdt));
			const size_t steps = boost::numeric::odeint::integrate_adaptive(controlled_stepper, harmonic_oscillator(gamma), x, 0.0, 10.0, 0.01);
			std::cout << "x = (" << x[0] << ',' << x[1] << "), step = " << steps << std::endl;
		}

		{
			state_type x(2);
			x[0] = 1.0;
			x[1] = 0.0;

			const size_t steps = boost::numeric::odeint::integrate_adaptive(boost::numeric::odeint::make_controlled<error_stepper_type>(1.0e-10, 1.0e-6), harmonic_oscillator(gamma), x, 0.0, 10.0, 0.01);
			std::cout << "x = (" << x[0] << ',' << x[1] << "), step = " << steps << std::endl;
		}

		//
		{
			state_type x(2);
			x[0] = 1.0;
			x[1] = 0.0;

			const size_t steps = boost::numeric::odeint::integrate_adaptive(boost::numeric::odeint::make_controlled(1.0e-10, 1.0e-6, error_stepper_type()), harmonic_oscillator(gamma), x, 0.0, 10.0, 0.01);
			std::cout << "x = (" << x[0] << ',' << x[1] << "), step = " << steps << std::endl;
		}
	}
}

}  // namespace local
}  // unnamed namespace

void odeint()
{
	local::harmonic_oscillator_example();

	// Use CUDA, OpenMP, TBB via Thrust.
	// Use OpenCL via VexCL.
	// Use parallel computation via OpenMP & MPI.
}
