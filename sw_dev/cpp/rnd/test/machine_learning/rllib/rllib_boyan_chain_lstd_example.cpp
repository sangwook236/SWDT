#include <rl.hpp>
#include <gsl/gsl_blas.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <vector>


namespace {
namespace local {

typedef rl::problem::boyan_chain::Simulator Simulator;

typedef Simulator::reward_type Reward;
typedef Simulator::observation_type State;
typedef Simulator::action_type Action;

struct Transition
{
	State s;
	Reward r;
	State s_;  // read s_ as s'.
	bool is_terminal;
};

std::ostream& operator<<(std::ostream& os, const Transition& t)
{
	os << t.s << " -- " << t.r;
	if (t.is_terminal)
		os << " End";
	else
		os << " --> " << t.s_;
	return os;
}

typedef std::vector<Transition> TransitionSet;

// The function that associates a feature vector to a State is the following.
typedef rl::problem::boyan_chain::Feature Feature;

// Let us define the parameters.
const double paramREG = 0;
const double paramGAMMA = 1;
const double paramALPHA = 0.05;

const int NB_OF_EPISODES = 100;

}  // namespace local
}  // unnamed namespace

namespace my_rllib {

// Least-squares temporal difference (LSTD).
// REF [paper] >> "Least-Squares Temporal Difference Learning", J. A. Boyan, ICML 1999.
// REF [paper] >> "Linear Least-Squares Algorithms for Temporal Difference Learning", S. J. Bradtke and A. G. Barto	, ML 1996.
// REF [file] >> ${RLlib_HOME}/examples/example-002-001-boyan-lstd.cc.
void boyan_chain_lstd_example()
{
	local::Simulator simulator;
	local::TransitionSet transitions;
	int episode_length;
	local::Feature phi;

	gsl_vector* theta = gsl_vector_alloc(phi.dimension());
	gsl_vector_set_zero(theta);
	gsl_vector* tmp = gsl_vector_alloc(phi.dimension());
	gsl_vector_set_zero(tmp);

	auto v_parametrized = [&phi, tmp](const gsl_vector* th, local::State s) -> local::Reward
	{
		double res;
		phi(tmp, s);  // phi_s = phi(s).
		gsl_blas_ddot(th, tmp, &res);  // res = th^T . phi_s.
		return res;
	};
	auto grad_v_parametrized = [&phi, tmp](const gsl_vector* th, gsl_vector* grad_th_s, local::State s) -> void
	{
		phi(tmp, s);  // phi_s = phi(s).
		gsl_vector_memcpy(grad_th_s, tmp);  // grad_th_s = phi_s.
	};

	try
	{
		// Fill a set of transitions from successive episodes.
		for (int episode = 0; episode < local::NB_OF_EPISODES; ++episode)
		{
			simulator.initPhase();
			rl::episode::run(
				simulator,
				[](local::State s) -> local::Action { return rl::problem::boyan_chain::actionNone; },  // This is the policy.
				std::back_inserter(transitions),
				[](local::State s, local::Action a, local::Reward r, local::State s_) -> local::Transition { return {s, r, s_,false}; },
				[](local::State s, local::Action a, local::Reward r) -> local::Transition { return {s, r, s ,true}; },
				0
			);
		}

		// Apply LSTD to the transition database.
		rl::lstd(
			theta,
			local::paramGAMMA, local::paramREG,
			transitions.begin(), transitions.end(),
			grad_v_parametrized,
			[](const local::Transition& t) -> local::State { return t.s; },
			[](const local::Transition& t) -> local::State { return t.s_; },
			[](const local::Transition& t) -> local::Reward { return t.r; },
			[](const local::Transition& t) -> bool { return t.is_terminal; }
		);

		// Display the result.
		std::cout << std::endl
			<< "LSTD estimation          : ("
			<< std::setw(15) << gsl_vector_get(theta, 0) << ','
			<< std::setw(15) << gsl_vector_get(theta, 1) << ','
			<< std::setw(15) << gsl_vector_get(theta, 2) << ','
			<< std::setw(15) << gsl_vector_get(theta, 3) << ')'
			<< std::endl;

		//Learn the same by using TD.
		auto td = rl::gsl::td<local::State>(
			theta,
			local::paramGAMMA, local::paramALPHA,
			v_parametrized,
			grad_v_parametrized
		);

		// The learning can be done offline since we have collected transitions.
		gsl_vector_set_zero(theta);
		for (auto& t : transitions)
			if (t.is_terminal)
				td.learn(t.s, t.r);
			else
				td.learn(t.s, t.r, t.s_);

		std::cout << "TD (offline) estimation  : ("
			<< std::setw(15) << gsl_vector_get(theta, 0) << ','
			<< std::setw(15) << gsl_vector_get(theta, 1) << ','
			<< std::setw(15) << gsl_vector_get(theta, 2) << ','
			<< std::setw(15) << gsl_vector_get(theta, 3) << ')'
			<< std::endl;

		// But this can be done on-line, directly from episodes.
		gsl_vector_set_zero(theta);
		for (int episode = 0; episode < local::NB_OF_EPISODES; ++episode)
		{
			simulator.initPhase();
			rl::episode::learn(
				simulator,
				[](local::State s) -> local::Action { return rl::problem::boyan_chain::actionNone; },  // This is the policy.
				td,
				0
			);
		}

		std::cout << "TD (online) estimation   : ("
			<< std::setw(15) << gsl_vector_get(theta, 0) << ','
			<< std::setw(15) << gsl_vector_get(theta, 1) << ','
			<< std::setw(15) << gsl_vector_get(theta, 2) << ','
			<< std::setw(15) << gsl_vector_get(theta, 3) << ')'
			<< std::endl;

		// With the boyan chain, the value function is known analytically.
		std::cout << "Optimal one should be    : ("
			<< std::setw(15) << -24 << ','
			<< std::setw(15) << -16 << ','
			<< std::setw(15) << -8 << ','
			<< std::setw(15) << 0 << ')'
			<< std::endl;
	}
	catch (const rl::exception::Any& e)
	{
		std::cerr << "Exception caught : " << e.what() << std::endl;
	}

	gsl_vector_free(theta);
	gsl_vector_free(tmp);
}

}  // namespace my_rllib
