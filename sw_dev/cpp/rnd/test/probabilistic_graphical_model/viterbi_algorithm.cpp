//#include "stdafx.h"
#include "viterbi.hpp"
#include <string>
#include <vector>
#include <map>
#include <iostream>

namespace {
namespace local_1 {

void viterbi_algorithm()
{
	// [ref] Viterbi Algorithm in C++ and using STL
	// http://bozskyfilip.blogspot.com/2009/01/viterbi-algorithm-in-c-and-using-stl.html

	Viterbi::HMM h;

	h.init();
	std::cout << h;

	Viterbi::forward_viterbi(
		h.get_observations(), 
		h.get_states(), 
		h.get_start_probability(), 
		h.get_transition_probability(), 
		h.get_emission_probability()
	);
}

}  // namespace local_1

namespace local_2 {

/*
	states = ('Rainy', 'Sunny')
 
	observations = ('walk', 'shop', 'clean')
 
	start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
 
	transition_probability = {
	   'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
	   'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
	}
 
	emission_probability = {
	   'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
	   'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
	}
*/

std::vector<std::string> states;
std::vector<std::string> observations;
std::map<std::string, double> start_probability;
std::map<std::string, std::map<std::string, double> > transition_probability;
std::map<std::string, std::map<std::string, double> > emission_probability;

class Tracking
{
public:
	double prob;
	std::vector<std::string> v_path;
	double v_prob;

	Tracking()
	{
		prob = 0.0;
		v_prob = 0.0;
	}

	Tracking(double p, std::vector<std::string> &v_pth, double v_p)
	{
		prob = p;
		v_path = v_pth;
		v_prob = v_p;
	}
};

void init_variables(void)
{
	states.push_back("Rainy");
	states.push_back("Sunny");

#if 0
	observations.push_back("walk");
	observations.push_back("shop");
	observations.push_back("clean");
	observations.push_back("walk");
	observations.push_back("shop");
	observations.push_back("clean");
	observations.push_back("clean");
	observations.push_back("clean");
#elif 1
	observations.push_back("clean");
	observations.push_back("shop");
	observations.push_back("walk");
	observations.push_back("walk");
	observations.push_back("clean");
	observations.push_back("shop");
#else
	observations.push_back("walk");
	observations.push_back("shop");
	observations.push_back("clean");
#endif

	start_probability["Rainy"] = 0.6;
	start_probability["Sunny"] = 0.4;

	transition_probability["Rainy"]["Rainy"] = 0.7;
	transition_probability["Rainy"]["Sunny"] = 0.3;
	transition_probability["Sunny"]["Rainy"] = 0.4;
	transition_probability["Sunny"]["Sunny"] = 0.6;

	emission_probability["Rainy"]["walk"] = 0.1;
	emission_probability["Rainy"]["shop"] = 0.4;
	emission_probability["Rainy"]["clean"] = 0.5;
	emission_probability["Sunny"]["walk"] = 0.6;
	emission_probability["Sunny"]["shop"] = 0.3;
	emission_probability["Sunny"]["clean"] = 0.1;
}

void print_variables(void)
{
	// print states
	std::cout << "States:" << std::endl;
	for (std::vector<std::string>::iterator i = states.begin(); i != states.end(); ++i)
	{
		std::cout << "S: " << (*i) << std::endl;
	}
	// print observations
	std::cout << "Observations:" << std::endl;
	for (std::vector<std::string>::iterator i = observations.begin();i!=observations.end();++i)
	{
		std::cout << "O: " << (*i) << std::endl;
	}

	// print start probabilities
	std::cout << "Start probabilities:" << std::endl;
	for (std::map<std::string, double>::iterator i = start_probability.begin(); i != start_probability.end(); ++i)
	{
		std::cout << "S: " << (*i).first << " P: " << (*i).second << std::endl;
	}

	// print transition_probability
	std::cout << "Transition probabilities:" << std::endl;
	for (std::map<std::string, std::map<std::string, double> >::iterator i = transition_probability.begin(); i != transition_probability.end(); ++i)
	{
		for (std::map<std::string, double>::iterator j = (*i).second.begin(); j != (*i).second.end(); ++j)
		{
			std::cout << "FS: " << (*i).first << " TS: " << (*j).first << " P: " << (*j).second << std::endl;
		}
	}

	// print emission probabilities
	std::cout << "Emission probabilities:" << std::endl;
	for (size_t i = 0; i < states.size(); ++i)
	{
		for (size_t j = 0; j < observations.size(); ++j)
		{
			std::cout << "FS: " << states[i] << " TO: " << observations[j] <<
				" P: " << emission_probability[states[i]][observations[j]] << std::endl;
		}
	}
}

//this method compute total probability for observation, most likely viterbi path 
//and probability of such path
void forward_viterbi(std::vector<std::string> obs, std::vector<std::string> states, std::map<std::string, double> start_p, 
                     std::map<std::string, std::map<std::string, double> > trans_p, 
                     std::map<std::string, std::map<std::string, double> > emit_p)
{
	std::map<std::string, Tracking> T;

	for (std::vector<std::string>::iterator state = states.begin(); state != states.end(); ++state)
	{
		std::vector<std::string> v_pth;
		v_pth.push_back(*state);

		T[*state] = Tracking(start_p[*state], v_pth, start_p[*state]);
	}

	for (std::vector<std::string>::iterator output = obs.begin(); output != obs.end(); ++output)
	{
		std::map<std::string, Tracking> U;

		for (std::vector<std::string>::iterator next_state = states.begin(); next_state != states.end(); ++next_state)
		{
			Tracking next_tracker;

			for (std::vector<std::string>::iterator source_state = states.begin(); source_state != states.end(); ++source_state)
			{
				Tracking source_tracker = T[*source_state];

				double p = emit_p[*source_state][*output]*trans_p[*source_state][*next_state];
				source_tracker.prob *= p;
				source_tracker.v_prob *= p;

				next_tracker.prob += source_tracker.prob;

				if (source_tracker.v_prob > next_tracker.v_prob)
				{
					next_tracker.v_path = source_tracker.v_path;
					next_tracker.v_path.push_back(*next_state);
					next_tracker.v_prob = source_tracker.v_prob;
				}
			}

			U[*next_state] = next_tracker;
		}

		T = U;
	}

	// apply sum/max to the final states
	Tracking final_tracker;

	for(std::vector<std::string>::iterator state = states.begin(); state != states.end(); ++state)
	{
		Tracking tracker = T[*state];

		final_tracker.prob += tracker.prob;

		if (tracker.v_prob > final_tracker.v_prob)
		{
			final_tracker.v_path = tracker.v_path;
			final_tracker.v_prob = tracker.v_prob;
		}
	}

	std::cout << "Total probability of the observation sequence: " << final_tracker.prob << std::endl;
	std::cout << "Probability of the Viterbi path: " << final_tracker.v_prob << std::endl;
	std::cout << "The Viterbi path: " << std::endl;
	for (std::vector<std::string>::iterator state = final_tracker.v_path.begin(); state != final_tracker.v_path.end(); ++state)
	{
		std::cout << "VState: " << *state << std::endl;
	}
}

void viterbi_algorithm()
{
	// [ref] Viterbi Algorithm in Boost and C++
	// http://codingplayground.blogspot.com/2009/02/viterbi-algorithm-in-boost-and-c.html

	init_variables();
	print_variables();

	forward_viterbi(
		observations, 
		states, 
		start_probability, 
		transition_probability, 
		emission_probability
	);
}

}  // namespace local_2
}  // unnamed namespace

void viterbi_algorithm()
{
	//local_1::viterbi_algorithm();
	local_2::viterbi_algorithm();
}
