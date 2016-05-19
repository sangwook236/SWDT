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

typedef rl::problem::mountain_car::DefaultParam mcParam;
typedef rl::problem::mountain_car::Simulator<mcParam> Simulator;

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

// This feature just transforms (s, a) into a vector (position, speed, action_is_None, action_is_Backward, action_is_Forward).
const int PHI_DIRECT_DIMENSION = 5;
void phi_direct(gsl_vector* phi, const State& s, const Action& a)
{
	gsl_vector_set_zero(phi);

	// We normalize position and speed
	gsl_vector_set(phi, 0, (s.position - Simulator::param_type::minPosition()) / (Simulator::param_type::maxPosition() - Simulator::param_type::minPosition()));
	gsl_vector_set(phi, 1, (s.speed - Simulator::param_type::minSpeed()) / (Simulator::param_type::maxSpeed() - Simulator::param_type::minSpeed()));
	switch (a)
	{
	case rl::problem::mountain_car::actionNone:
		gsl_vector_set(phi, 2, 1.0);
		break;
	case rl::problem::mountain_car::actionBackward:
		gsl_vector_set(phi, 3, 1.0);
		break;
	case rl::problem::mountain_car::actionForward:
		gsl_vector_set(phi, 4, 1.0);
		break;
	default:
		throw rl::problem::mountain_car::BadAction(" in phi_direct()");
	}
}

// This is the feature based on gaussian RBF.
const int SPLIT = 5;
const double SIGMA = 1 / (SPLIT - 1.0);
const double SIGMA2 = SIGMA * SIGMA;

const double DELTA_POSITION = Simulator::param_type::maxPosition() - Simulator::param_type::minPosition();
const double DELTA_POSITION2 = DELTA_POSITION * DELTA_POSITION;

const double DELTA_SPEED = Simulator::param_type::maxSpeed() - Simulator::param_type::minSpeed();
const double DELTA_SPEED2 = DELTA_SPEED * DELTA_SPEED;

const int PHI_RBF_DIMENSION = (SPLIT * SPLIT + 1) * 3;
class RBFFeature
{
public:
	RBFFeature(void)
	{
		// These are the centers of the gaussian that spans State.
		double position_step = DELTA_POSITION / (SPLIT - 1.0);
		double speed_step = DELTA_SPEED / (SPLIT - 1.0);

		for (int i = 0; i < SPLIT; ++i)
			position[i] = i * position_step + Simulator::param_type::minPosition();

		for (int i = 0; i < SPLIT; ++i)
			speed[i] = i * speed_step + Simulator::param_type::minSpeed();
	}

	void operator()(gsl_vector *phi, const State&  s, const Action& a) const
	{

		if ((gsl_vector*)0 == phi)
			throw rl::exception::NullVectorPtr("in Feature::operator()");
		else if (PHI_RBF_DIMENSION != (int)(phi->size))
			throw rl::exception::BadVectorSize(phi->size, PHI_RBF_DIMENSION, "in Feature::operator()");

		int action_offset;
		switch (a)
		{
		case rl::problem::mountain_car::actionNone:
			action_offset = 0;
			break;
		case rl::problem::mountain_car::actionBackward:
			action_offset = SPLIT * SPLIT + 1;
			break;
		case rl::problem::mountain_car::actionForward:
			action_offset = 2 * (SPLIT * SPLIT + 1);
			break;
		default:
			throw rl::problem::inverted_pendulum::BadAction("in Feature::operator()");
		}

		double dposition, dspeed;
		gsl_vector_set_zero(phi);
		for (int i = 0, k = action_offset + 1; i < SPLIT; ++i)
		{
			dposition = s.position - position[i];
			dposition *= dposition;
			dposition /= 2 * SIGMA2 * DELTA_POSITION2;
			for (int j = 0; j < SPLIT; ++j, ++k)
			{
				dspeed = s.speed - speed[j];
				dspeed *= dspeed;
				dspeed /= 2 * SIGMA2 * DELTA_SPEED2;
				gsl_vector_set(phi, k, std::exp(-dposition - dspeed));
			}
			gsl_vector_set(phi, action_offset, 1);
		}
	}

private:
	double position[SPLIT];
	double speed[SPLIT];
};

// Define the parameters.
const double paramGAMMA = 0.95;
const double paramEPSILON = .1;
const double paramETA_NOISE = 1e-5;
const double paramOBSERVATION_NOISE = 1;
const double paramPRIOR_VAR = 10;
const double paramRANDOM_AMPLITUDE = 1e-1;
const double paramUT_ALPHA = 1e-1;
const double paramUT_BETA = 2;
const double paramUT_KAPPA = 0;
const bool paramUSE_LINEAR_EVALUATION = true;  // Use a linear architecture.

typedef rl::problem::mountain_car::Gnuplot<Simulator>  Gnuplot;

const int MAX_EPISODE_LENGTH_LEARN = 1500;
const int MAX_EPISODE_LENGTH_TEST = 300;
#define KTDSARSA_FILENAME   "data/mountain-car.ktdsarsa"

void train(int nb_episodes, bool make_movie)
{
	RBFFeature phi;

	gsl_vector* theta = gsl_vector_alloc(PHI_RBF_DIMENSION);
	gsl_vector_set_zero(theta);
	gsl_vector* tmp = gsl_vector_alloc(PHI_RBF_DIMENSION);
	gsl_vector_set_zero(tmp);

	auto q_parametrized = [tmp, &phi](const gsl_vector* th, State s, Action a) -> Reward
	{
		double res;
		phi(tmp, s, a);  // phi_sa = phi(s, a).
		gsl_blas_ddot(th, tmp, &res); // res = th^T . phi_sa.
		return res;
	};

	auto q = std::bind(q_parametrized, theta, std::placeholders::_1, std::placeholders::_2);

	rl::enumerator<Action> a_begin(rl::problem::inverted_pendulum::actionNone);
	rl::enumerator<Action> a_end = a_begin + 3;

	auto explore_agent = rl::policy::epsilon_greedy(q, paramEPSILON, a_begin, a_end);
	auto greedy_agent = rl::policy::greedy(q, a_begin, a_end);

	auto critic = rl::gsl::ktd_sarsa<State, Action>(
		theta,
		q_parametrized,
		paramGAMMA,
		paramETA_NOISE,
		paramOBSERVATION_NOISE,
		paramPRIOR_VAR,
		paramRANDOM_AMPLITUDE,
		paramUT_ALPHA,
		paramUT_BETA,
		paramUT_KAPPA,
		paramUSE_LINEAR_EVALUATION
	);

	// Let us initialize the random seed.
#if defined(_WIN64) || defined(_WIN32)
		rl::random::seed((unsigned int)std::time(NULL));
#else
		rl::random::seed(getpid());
#endif

	try
	{
		Simulator simulator;
		int step = 0, episode_length;
		for (int episode = 0; episode < nb_episodes; ++episode)
		{
			std::cout << "Running episode " << episode + 1 << "/" << nb_episodes << "." << std::endl;
			simulator.setPhase(Simulator::phase_type());
			episode_length = rl::episode::learn(simulator, explore_agent, critic, MAX_EPISODE_LENGTH_LEARN);
			std::cout << "... length is " << episode_length << "." << std::endl;

			++step;

			if (make_movie)
				Gnuplot::drawQ("KTD Sarsa + RBF", "ktd", step, critic, greedy_agent);
		}

		// Let us save the results.
		std::ofstream file("data/mountain-car.ktdsarsa");
		if (!file)
			std::cerr << "Cannot open \"" << "data/mountain-car.ktdsarsa" << "\"." << std::endl;
		else
		{
			file << std::setprecision(20) << critic;
			file.close();
		}

		if (make_movie)
		{
			std::string command;

			command = "find . -name \"ktd-*.plot\" -exec gnuplot \\{} \\;";
			std::cout << "Executing : " << command << std::endl;
			system(command.c_str());

			command = "find . -name \"ktd-*.png\" -exec convert \\{} -quality 100 \\{}.jpg \\;";
			std::cout << "Executing : " << command << std::endl;
			system(command.c_str());

			command = "ffmpeg -i ktd-%06d.png.jpg -b 1M rllib.avi";
			std::cout << "Executing : " << command << std::endl;
			system(command.c_str());

			command = "find . -name \"ktd-*.plot\" -exec rm \\{} \\;";
			std::cout << "Executing : " << command << std::endl;
			system(command.c_str());

			command = "find . -name \"ktd-*.png\" -exec rm \\{} \\;";
			std::cout << "Executing : " << command << std::endl;
			system(command.c_str());

			command = "find . -name \"ktd-*.png.jpg\" -exec rm \\{} \\;";
			std::cout << "Executing : " << command << std::endl;
			system(command.c_str());
		}
	}
	catch (const rl::exception::Any& e)
	{
		std::cerr << "Exception caught : " << e.what() << std::endl;
	}
}

void test(const Simulator::phase_type& start)
{
	RBFFeature phi;

	gsl_vector* theta = gsl_vector_alloc(PHI_RBF_DIMENSION);
	gsl_vector_set_zero(theta);
	gsl_vector* tmp = gsl_vector_alloc(PHI_RBF_DIMENSION);
	gsl_vector_set_zero(tmp);

	auto q_parametrized = [tmp, &phi](const gsl_vector* th, State s, Action a) -> Reward
	{
		double res;
		phi(tmp, s, a);  // phi_sa = phi(s, a).
		gsl_blas_ddot(th, tmp, &res);  // res = th^T . phi_sa.
		return res;
	};

	auto q = std::bind(q_parametrized, theta, std::placeholders::_1, std::placeholders::_2);

	rl::enumerator<Action> a_begin(rl::problem::inverted_pendulum::actionNone);
	rl::enumerator<Action> a_end = a_begin + 3;

	auto greedy_agent = rl::policy::greedy(q, a_begin, a_end);

	auto critic = rl::gsl::ktd_sarsa<State, Action>(
		theta,
		q_parametrized,
		paramGAMMA,
		paramETA_NOISE,
		paramOBSERVATION_NOISE,
		paramPRIOR_VAR,
		paramRANDOM_AMPLITUDE,
		paramUT_ALPHA,
		paramUT_BETA,
		paramUT_KAPPA,
		paramUSE_LINEAR_EVALUATION
	);

	try
	{
		std::ifstream file("data/mountain-car.ktdsarsa");
		if (!file)
		{
			std::cerr << "Cannot open \"" << "data/mountain-car.ktdsarsa" << "\"." << std::endl;
			return;
		}

		// Load some critic.
		file >> critic;

		// Run an episode.
		Simulator simulator;
		simulator.setPhase(start);
		Gnuplot::drawEpisode("Mountain car run", "mountain-car-run", -1, simulator, critic, greedy_agent, MAX_EPISODE_LENGTH_TEST);
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
// REF [file] >> ${RLlib_HOME}/examples/example-003-003-mountain-car-ktdsarsa.cc.
void mountain_car_ktdsqrsa_example()
{
	const bool learn_mode = true;
	const bool movie_mode = false;
	const int nb_episodes = 100;
	local::Simulator simulator;
	local::Simulator::phase_type init_phase = local::Simulator::phase_type(simulator.bottom(), 0);  // bottom
	//local::Simulator::phase_type init_phase = local::Simulator::phase_type();  // random.
	//local::Simulator::phase_type init_phase = local::Simulator::phase_type(10.0, 0.5);  // (position, speed).

#if defined(_WIN64) || defined(_WIN32)
	rl::random::seed((unsigned int)std::time(NULL));
#else
	rl::random::seed(getpid());
#endif

	if (learn_mode)
		local::train(nb_episodes, movie_mode);
	else
		local::test(init_phase);
}

}  // namespace my_rllib
