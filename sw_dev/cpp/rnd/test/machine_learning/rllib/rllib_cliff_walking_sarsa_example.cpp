#include <rl.hpp>
#include <iostream>
#include <array>
#include <functional>
#include <algorithm>  
#include <vector>
#include <cstdlib>  


namespace {
namespace local {

typedef rl::problem::cliff_walking::Cliff<20, 6> Cliff;
typedef rl::problem::cliff_walking::Param Param;
typedef rl::problem::cliff_walking::Simulator<Cliff, Param> Simulator;

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

const size_t S_CARDINALITY = Cliff::size;
const size_t A_CARDINALITY = rl::problem::cliff_walking::actionSize;
const size_t TABULAR_Q_CARDINALITY = S_CARDINALITY * A_CARDINALITY;
size_t TABULAR_Q_RANK(State s, Action a) { return a * S_CARDINALITY + s; }

// Let us define the parameters.
const double paramGAMMA = 0.99;
const double paramALPHA = 0.05;
const double paramEPSILON = 0.2;

double q_parametrized(const gsl_vector* theta, State s, Action a)
{
	return gsl_vector_get(theta, TABULAR_Q_RANK(s, a));
}

void grad_q_parametrized(const gsl_vector* theta, gsl_vector* grad_theta_sa, State s, Action a)
{
	gsl_vector_set_basis(grad_theta_sa, TABULAR_Q_RANK(s, a));
}

const int NB_EPISODES = 10000;
const int MAX_EPISODE_DURATION = 100;
const int FRAME_PERIOD = 25;
const double MIN_V = -50;

// This is an output iterator that notifies the visited states.
// The use within the run function is like
//
// VisitNotifier v;
//
// *(v++) = transition(s,a,r,s');
//
// Here, we will use transition(s,a,r,s') = s, see the lambda
// functions given to the run function.
class VisitNotifier
{
public:
	VisitNotifier(std::array<bool, Cliff::size>& v)
		: visited(v)
	{
		std::fill(visited.begin(), visited.end(), false);
	}

	VisitNotifier(const VisitNotifier& cp) : visited(cp.visited) {}

	VisitNotifier& operator*() { return *this; }
	VisitNotifier& operator++(int) { return *this; }

	void operator=(State s)
	{
		visited[s] = true;
	}

public:
	std::array<bool, Cliff::size>& visited;
};

template<typename CRITIC, typename Q>
void make_experiment(CRITIC& critic, const Q& q)
{
	Param param;
	Simulator simulator(param);
	auto action_begin = rl::enumerator<Action>(rl::problem::cliff_walking::actionNorth);
	auto action_end = action_begin + rl::problem::cliff_walking::actionSize;
	auto state_begin = rl::enumerator<State>(Cliff::start);
	auto state_end = state_begin + Cliff::size;
	auto learning_policy = rl::policy::epsilon_greedy(q, paramEPSILON, action_begin, action_end);
	auto test_policy = rl::policy::greedy(q, action_begin, action_end);
	int episode, frame;
	int episode_length;

	std::array<bool, Cliff::size> visited;

	std::cout << std::endl << std::endl;
	for (episode = 0, frame = 0; episode < NB_EPISODES; ++episode)
	{
		std::cout << "running episode " << std::setw(6) << episode + 1 << "/" << NB_EPISODES << "    \r" << std::flush;

		simulator.restart();
		auto actual_episode_length = rl::episode::learn(simulator, learning_policy, critic, 0);

		if (episode % FRAME_PERIOD == 0)
		{
			// Let us run an episode with a greedy policy and mark the states as visited.
			VisitNotifier visit_notifier(visited);
			simulator.restart();
			rl::episode::run(
				simulator,
				test_policy,
				visit_notifier,
				[](State s, Action a, Reward r, State s_) -> State { return s; },
				[](State s, Action a, Reward r) -> State { return s; },
				MAX_EPISODE_DURATION
			);

			Cliff::draw_visited(
				"data/rllib-cliff-sarsa", frame++,
				[&action_begin, &action_end, &q](State s) -> double { return rl::max(std::bind(q, s, std::placeholders::_1), action_begin, action_end); },  // V(s) = max_a q(s,q)
				[&visit_notifier](State s) -> bool { return visit_notifier.visited[s]; },
				MIN_V, 0
			);
		}
	}

	std::cout << std::endl << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_rllib {

// SARSA.
// REF [file] >> ${RLlib_HOME}/examples/example-001-001-cliff-walking-sarsa.cc.
void cliff_walking_sarsa_example()
{
	gsl_vector* theta = gsl_vector_alloc(local::TABULAR_Q_CARDINALITY);
	gsl_vector_set_zero(theta);

	auto critic = rl::gsl::sarsa<local::State, local::Action>(
		theta,
		local::paramGAMMA, local::paramALPHA,
		local::q_parametrized,
		local::grad_q_parametrized
	);

	auto q = std::bind(local::q_parametrized, theta, std::placeholders::_1, std::placeholders::_2);
	local::make_experiment(critic, q);

	gsl_vector_free(theta);
}

}  // namespace my_rllib
