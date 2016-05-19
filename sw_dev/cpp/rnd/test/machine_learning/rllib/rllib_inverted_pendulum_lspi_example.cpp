#if defined(_MSC_VER)
#define _USE_MATH_DEFINES
#include <math.h>
#endif
#include <rl.hpp>
#include <gsl/gsl_vector.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <functional>
#include <array>
#include <cmath>
#if defined(_WIN64) || defined(_WIN32)
#include <ctime>
#else
#include <unistd.h>
#endif


namespace {
namespace local {

// This is our simulator.
typedef rl::problem::inverted_pendulum::Simulator<rl::problem::inverted_pendulum::DefaultParam> Simulator;

typedef Simulator::reward_type Reward;
typedef Simulator::observation_type State;
typedef Simulator::action_type Action;

struct Transition
{
	State s;
	Action a;
	Reward r;
	State s_;  // read s_ as s'.
	Action a_;  // read a_ as a'.
	bool is_terminal;
};

typedef std::vector<Transition> TransitionSet;

// Here are general purpose setting and reading functions.
Transition make_transition(State s, Action a, Reward r, State s_) { return {s, a, r, s_, a /* unused */, false}; }
Transition make_terminal_transition(State s, Action a, Reward r) { return {s, a, r, s /* unused */, a /* unused */, true}; }
State next_state_of(const Transition& t) { return t.s_; }
Reward reward_of(const Transition& t) { return t.r; }
bool is_terminal(const Transition& t) { return t.is_terminal; }
void set_next_action(Transition& t, Action a) { t.a_ = a; }

rl::sa::Pair<State, Action> current_of(const Transition& t) { return {t.s, t.a}; }
rl::sa::Pair<State, Action> next_of(const Transition& t) { return {t.s_, t.a_}; }
Transition make_transition_sa(const rl::sa::Pair<State, Action>& z, Reward r, const rl::sa::Pair<State, Action>& z_) { return {z.s, z.a, r, z_.s, z_.a, false}; }
Transition make_terminal_transition_sa(const rl::sa::Pair<State, Action>& z, Reward r) { return {z.s, z.a, r, z.s /* unused */, z.a /* unused */, true}; }

// This feature justs transform (s,a) into a vector (angle, speed, action_is_None, action_is_Left, action_is_Right).
const int PHI_DIRECT_DIMENSION = 5;
void phi_direct(gsl_vector* phi, const State& s, const Action& a)
{
	gsl_vector_set_zero(phi);
	gsl_vector_set(phi, 0, s.angle);
	gsl_vector_set(phi, 1, s.speed);
	switch (a)
	{
	case rl::problem::inverted_pendulum::actionNone:
		gsl_vector_set(phi, 2, 1.0);
		break;
	case rl::problem::inverted_pendulum::actionLeft:
		gsl_vector_set(phi, 3, 1.0);
		break;
	case rl::problem::inverted_pendulum::actionRight:
		gsl_vector_set(phi, 4, 1.0);
		break;
	default:
		throw rl::problem::inverted_pendulum::BadAction(" in phi_direct()");
	}
}

const int PHI_RBF_DIMENSION = 30;
void phi_rbf(gsl_vector* phi, const State& s, const Action& a)
{
	std::array<double, 3> angle = { { -M_PI_4, 0, M_PI_4 } };
	std::array<double, 3> speed = { { -1, 0, 1 } };

	if ((gsl_vector*)0 == phi)
		throw rl::exception::NullVectorPtr("in Feature::operator()");
	else if (PHI_RBF_DIMENSION != (int)(phi->size))
		throw rl::exception::BadVectorSize(phi->size, PHI_RBF_DIMENSION, "in Feature::operator()");

	int action_offset;
	switch (a)
	{
	case rl::problem::inverted_pendulum::actionNone:
		action_offset = 0;
		break;
	case rl::problem::inverted_pendulum::actionLeft:
		action_offset = 10;
		break;
	case rl::problem::inverted_pendulum::actionRight:
		action_offset = 20;
		break;
	default:
		throw rl::problem::inverted_pendulum::BadAction("in phi_rbf()");
	}

	double dangle, dspeed;
	gsl_vector_set_zero(phi);
	for (int i = 0, k = action_offset + 1; i < 3; ++i)
	{
		dangle = s.angle - angle[i];
		dangle *= dangle;
		for (int j = 0; j < 3; ++j, ++k)
		{
			dspeed = s.speed - speed[j];
			dspeed *= dspeed;
			gsl_vector_set(phi, k, std::exp(-.5 * (dangle + dspeed)));
		}
		gsl_vector_set(phi, action_offset, 1);
	}
}

// Define the parameters.
const double paramREG = 0;
const double paramGAMMA = 0.95;

const int NB_OF_EPISODES = 1000;
const int NB_ITERATION_STEPS = 10;
const int MAX_EPISODE_LENGTH = 3000;
const int NB_LENGTH_SAMPLES = 20;

template<typename POLICY>
void test_iteration(const POLICY& policy, int step)
{
	Simulator simulator;

	int length;
	double mean_length = 0;
	for (int episode = 0; episode < NB_LENGTH_SAMPLES; ++episode)
	{
		// Generate an episode and get its length.
		simulator.setPhase(Simulator::phase_type());
		length = rl::episode::run(simulator, policy, MAX_EPISODE_LENGTH);

		// Display the length.
		std::cout << "\rStep " << std::setw(4) << std::setfill('0') << step
			<< " : " << std::setfill('.') << std::setw(4) << episode + 1 << " length = "
			<< std::setw(10) << std::setfill(' ')
			<< length << std::flush;

		// Update mean.
		mean_length += length;
	}

	mean_length /= NB_LENGTH_SAMPLES;
	std::cout << "\rStep "
		<< std::setw(4) << std::setfill('0') << step
		<< " : mean length = "
		<< std::setw(10) << std::setfill(' ')
		<< .01 * (int)(mean_length * 100 + .5) << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_rllib {

// Least-squares policy iteration (LSPI).
// REF [paper] >> "Least-Squares Policy Iteration", M. G. Lagoudakis and R. Parr, JMLR 2003.
// REF [file] >> ${RLlib_HOME}/examples/example-002-002-pendulum-lspi.cc.
void inverted_pendulum_lspi_example()
{
	gsl_vector* theta = gsl_vector_alloc(local::PHI_RBF_DIMENSION);
	gsl_vector_set_zero(theta);
	gsl_vector* tmp = gsl_vector_alloc(local::PHI_RBF_DIMENSION);
	gsl_vector_set_zero(tmp);

	auto q_parametrized = [tmp](const gsl_vector* th, local::State s, local::Action a) -> local::Reward
	{
		double res;
		local::phi_rbf(tmp, s, a);  // phi_sa = phi(s, a).
		gsl_blas_ddot(th, tmp, &res);  // res = th^T . phi_sa.
		return res;
	};
	auto grad_q_parametrized = [tmp](const gsl_vector* th, gsl_vector* grad_th_s, local::State s, local::Action a) -> void
	{
		local::phi_rbf(tmp, s, a);  // phi_sa = phi(s, a).
		gsl_vector_memcpy(grad_th_s, tmp);  // grad_th_s = phi_sa.
	};

	rl::enumerator<local::Action> a_begin(rl::problem::inverted_pendulum::actionNone);
	rl::enumerator<local::Action> a_end = a_begin + 3;
	auto random_policy = rl::policy::random(a_begin, a_end);
	auto q = std::bind(q_parametrized, theta, std::placeholders::_1, std::placeholders::_2);
	auto greedy_policy = rl::policy::greedy(q, a_begin, a_end);

	try
	{
		local::Simulator simulator;
		local::TransitionSet transitions;

		// Initialize the random seed.
#if defined(_WIN64) || defined(_WIN32)
		rl::random::seed((unsigned int)std::time(NULL));
#else
		rl::random::seed(getpid());
#endif

		// Fill a set of transitions from successive episodes, using a random policy.
		for (int episode = 0; episode < local::NB_OF_EPISODES; ++episode)
		{
			simulator.setPhase(local::Simulator::phase_type());
			rl::episode::run(
				simulator,
				random_policy,
				std::back_inserter(transitions),
				local::make_transition,
				local::make_terminal_transition,
				0
			);
		}

		// Try the random policy.
		local::test_iteration(random_policy, 0);

		// Let us used LSTD as a batch critic. LSTD considers transitions
		// as (Z,r,Z'), but Z is a pair (s,a) here. See the definitions of
		// current_of, next_of, reward_of, and note that
		// gradvparam_of_gradqparam transforms Q(s,a) into V(z).
		auto critic = [theta, grad_q_parametrized](const local::TransitionSet::iterator& t_begin, const local::TransitionSet::iterator& t_end) -> void
		{
			rl::lstd(
				theta, local::paramGAMMA, local::paramREG,
				t_begin, t_end,
				rl::sa::gsl::gradvparam_of_gradqparam<local::State, local::Action, local::Reward>(grad_q_parametrized),
				local::current_of, local::next_of, local::reward_of, local::is_terminal
			);
		};

		// Improve the policy and measure its performance at each step.
		for (int step = 1; step <= local::NB_ITERATION_STEPS; ++step)
		{
			rl::batch_policy_iteration_step(
				critic, q,
				transitions.begin(), transitions.end(),
				a_begin, a_end,
				local::is_terminal, local::next_state_of, local::set_next_action
			);
			local::test_iteration(greedy_policy, step);
		}

		// Save the q_theta parameter.
		std::cout << "Writing lspi.data" << std::endl;
		std::ofstream ofile("data/lspi.data");
		if (!ofile)
			std::cerr << "cannot open file for writing" << std::endl;
		else
		{
			ofile << theta << std::endl;
			ofile.close();
		}
	}
	catch (const rl::exception::Any& e)
	{
		std::cerr << "Exception caught : " << e.what() << std::endl;
	}
}

}  // namespace my_rllib
