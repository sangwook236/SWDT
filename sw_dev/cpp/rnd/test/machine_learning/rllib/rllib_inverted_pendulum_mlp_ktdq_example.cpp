#if defined(_MSC_VER)
#define _USE_MATH_DEFINES
#include <math.h>
#endif
#include <rl.hpp>
#include <gsl/gsl_blas.h>
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

// define our own parameters for the inverted pendulum.
class ipParams
{
public:
	// This is the amplitude of the noise (relative) applied to the action.
	inline static double actionNoise(void)  { return  0.2; }
	// This is the noise of angle perturbation from the equilibrium state at initialization.
	inline static double angleInitNoise(void)  { return 1e-3; }
	// This is the noise of speed perturbation from the equilibrium state at initialization.
	inline static double speedInitNoise(void)  { return 1e-3; }
};

// This is our simulator.
typedef rl::problem::inverted_pendulum::Simulator<ipParams> Simulator;

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

// This feature justs transform (s, a) into a vector (angle, speed, action_is_None, action_is_Left, action_is_Right).
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
		throw rl::exception::NullVectorPtr("in phi_rbf()");
	else if (PHI_RBF_DIMENSION != (int)(phi->size))
		throw rl::exception::BadVectorSize(phi->size, PHI_RBF_DIMENSION, "in phi_rbf()");

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
const double paramGAMMA = 0.95;
const double paramSIGMOID_COEF = .1;
const double paramETA_NOISE = 0;
const double paramOBSERVATION_NOISE = 1e-4;
const double paramPRIOR_VAR = std::sqrt(1e-1);
const double paramRANDOM_AMPLITUDE = 1e-1;
const double paramUT_ALPHA = 1e-2;
const double paramUT_BETA = 2;
const double paramUT_KAPPA = 0;
const bool paramUSE_LINEAR_EVALUATION = false;

const int NB_OF_EPISODES = 1000;
const int NB_LENGTH_SAMPLES = 5;
const int MAX_EPISODE_LENGTH = 3000;
const int TEST_PERIOD = 100;

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

template<typename CRITIC, typename fctQ, typename ACTION_ITERATOR>
void make_experiment(CRITIC& critic, const fctQ& q, const ACTION_ITERATOR& a_begin, const ACTION_ITERATOR& a_end)
{
	Simulator simulator;
	CRITIC critic_loaded = critic;

	auto explore_agent = rl::policy::random(a_begin, a_end);
	auto greedy_agent = rl::policy::greedy(q, a_begin, a_end);

	int step;
	try
	{
		step = 0;

		// Initialize the random seed.
#if defined(_WIN64) || defined(_WIN32)
		rl::random::seed((unsigned int)std::time(NULL));
#else
		rl::random::seed(getpid());
#endif

		for (int episode = 0; episode < NB_OF_EPISODES; ++episode)
		{
			simulator.setPhase(Simulator::phase_type());
			rl::episode::learn(simulator, explore_agent, critic, MAX_EPISODE_LENGTH);
			if (0 == (episode % TEST_PERIOD))
			{
				++step;
				test_iteration(greedy_agent, step);
			}
		}

		// Save the ktdq object.
		std::cout << "Writing ktdq.data" << std::endl;
		std::ofstream ofile("data/ktdq.data");
		if (!ofile)
			std::cerr << "cannot open file for writing" << std::endl;
		else
		{
			ofile << critic;
			ofile.close();
		}

		// Load back with >>
		std::cout << "Reading ktdq.data" << std::endl;
		std::ifstream ifile("data/ktdq.data");
		if (!ifile)
			std::cerr << "cannot open file for reading" << std::endl;
		else
		{
			ifile >> critic_loaded;
			ifile.close();
		}

		// As the theta parameter is shared by q and the critic, the load of the critic modifies q, and thus the greedy agent.

		// Try this loaded ktdq.
		test_iteration(greedy_agent, step);
	}
	catch (const rl::exception::Any& e)
	{
		std::cerr << "Exception caught : " << e.what() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_rllib {

// Kalman temporal difference (KTD).
// REF [file] >> ${RLlib_HOME}/examples/example-003-002-pendulum-mlp-ktdq.cc.
void inverted_pendulum_mlp_ktdq_example()
{
	// Setup the Q-function approximator as a perceptron.
	auto sigmoid = std::bind(rl::transfer::tanh, std::placeholders::_1, local::paramSIGMOID_COEF);
	auto input_layer = rl::gsl::mlp::input<local::State, local::Action>(local::phi_direct, local::PHI_DIRECT_DIMENSION);
	auto hidden_layer_1 = rl::gsl::mlp::hidden(input_layer, 5, sigmoid);
	auto hidden_layer_2 = rl::gsl::mlp::hidden(hidden_layer_1, 3, sigmoid);
	auto q_parametrized = rl::gsl::mlp::output(hidden_layer_2, rl::transfer::identity);

	gsl_vector* theta = gsl_vector_alloc(q_parametrized.size);
	gsl_vector_set_zero(theta);

	// Display the structure of our MLP...
	std::cout << std::endl;
	q_parametrized.displayParameters(std::cout);
	std::cout << std::endl;

	auto q = std::bind(q_parametrized, theta, std::placeholders::_1, std::placeholders::_2);

	rl::enumerator<local::Action> a_begin(rl::problem::inverted_pendulum::actionNone);
	rl::enumerator<local::Action> a_end = a_begin + 3;

	auto critic = rl::gsl::ktd_q<local::State, local::Action>(
		theta,
		q_parametrized,
		a_begin, a_end,
		local::paramGAMMA,
		local::paramETA_NOISE,
		local::paramOBSERVATION_NOISE,
		local::paramPRIOR_VAR,
		local::paramRANDOM_AMPLITUDE,
		local::paramUT_ALPHA,
		local::paramUT_BETA,
		local::paramUT_KAPPA,
		local::paramUSE_LINEAR_EVALUATION
	);

	local::make_experiment(critic, q, a_begin, a_end);
}

}  // namespace my_rllib
