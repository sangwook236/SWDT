//#include "stdafx.h"
#include "viterbi.hpp"

namespace Viterbi {

  void HMM::init(void) {
    states__.push_back("Rainy");
    states__.push_back("Sunny");

#if 0
	observations__.push_back("walk");
    observations__.push_back("shop");
    observations__.push_back("clean");
    observations__.push_back("walk");
    observations__.push_back("shop");
    observations__.push_back("clean");
    observations__.push_back("clean");
    observations__.push_back("clean");
#elif 1
	observations__.push_back("clean");
    observations__.push_back("shop");
    observations__.push_back("walk");
    observations__.push_back("walk");
    observations__.push_back("clean");
    observations__.push_back("shop");
#else
	observations__.push_back("walk");
    observations__.push_back("shop");
    observations__.push_back("clean");
#endif
	
	start_probability__["Rainy"] = 0.6;
    start_probability__["Sunny"] = 0.4;

    transition_probability__["Rainy"]["Rainy"] = 0.7;
    transition_probability__["Rainy"]["Sunny"] = 0.3;
    transition_probability__["Sunny"]["Rainy"] = 0.4;
    transition_probability__["Sunny"]["Sunny"] = 0.6;

    emission_probability__["Rainy"]["walk"] = 0.1;
    emission_probability__["Rainy"]["shop"] = 0.4;
    emission_probability__["Rainy"]["clean"] = 0.5;
    emission_probability__["Sunny"]["walk"] = 0.6;
    emission_probability__["Sunny"]["shop"] = 0.3;
    emission_probability__["Sunny"]["clean"] = 0.1;
  }


  std::ostream& operator << (std::ostream& os, HMM& h){
    
    os << "States:" << std::endl;
    BOOST_FOREACH(Seq_States::value_type s, h.states__){
      os << "\tS: " << s << std::endl;
    }
    
    os << "Observations:" << std::endl;
    BOOST_FOREACH(Seq_States::value_type s, h.observations__){
      os << "\tO: " << s << std::endl;
    }
    
    os << "Start probabilities:" << std::endl;
    BOOST_FOREACH(Map_State_Probability::value_type m, 
		  h.start_probability__){
      os << "\tS: " << m.first 
	 << " P: " << m.second << std::endl;
    }
    
    os << "Transition probabilities:" << std::endl;
    BOOST_FOREACH(State_Map_State_Probability::value_type tm,
		  h.transition_probability__){
      BOOST_FOREACH(Map_State_Probability::value_type m,
		    tm.second){
	
	  os << "\t FS: " << tm.first << " TS: " << m.first 
	     << " P: " << m.second << std::endl;
      }
    }
    
    os << "Emission probabilities:" << std::endl;
    BOOST_FOREACH(State_Map_State_Probability::value_type em,
		  h.emission_probability__){
      BOOST_FOREACH(Map_State_Probability::value_type m,
		      em.second){
	
	os << "\tFS: " << em.first << " TO: " << m.first 
	   << " P: " << m.second << std::endl;
      }
    }
    return os;
  };

 
  //
  // supporting structure
  //
  struct Tracking {
    Probability prob;
    Seq_States v_path;
    Probability v_prob;
    
    Tracking() : prob(0.0), v_prob(0.0) {}; 
    Tracking(const Probability p, 
	     Seq_States& v_pth, 
	     const Probability v_p): 
      prob(p), v_path(v_pth), v_prob(v_p) {};
  };


  // computes total probability for observation
  // most likely viterbi path 
  // and probability of such path
  void forward_viterbi(const Seq_States &obs, 
		       const Seq_States &states, 
		       Map_State_Probability &start_p, 
		       State_Map_State_Probability &trans_p, 
		       State_Map_State_Probability &emit_p){

    typedef boost::unordered_map<State, Tracking> Tracker_Map;
    Tracker_Map T;

    BOOST_FOREACH(Seq_States::value_type s, states){

      Seq_States v_pth;
      v_pth.push_back(s);
      T[s] = Tracking(start_p[s], v_pth, start_p[s]);
    }

    BOOST_FOREACH(Seq_States::value_type output, obs){

      Tracker_Map U;

      BOOST_FOREACH(Seq_States::value_type next_state, states){

	Tracking next_tracker;
	
	BOOST_FOREACH(Seq_States::value_type source_state, states){

	  Tracking source_tracker = T[source_state];
	  
	  Probability p = emit_p[source_state][output]*
	    trans_p[source_state][next_state];
	  source_tracker.prob *= p;
	  source_tracker.v_prob *= p;
	  
	  next_tracker.prob += source_tracker.prob;

	  if(source_tracker.v_prob > next_tracker.v_prob) {
	    next_tracker.v_path = source_tracker.v_path;
	    next_tracker.v_path.push_back(next_state);
	    next_tracker.v_prob = source_tracker.v_prob;
	  }
	}
	U[next_state] = next_tracker;
      }
      T = U;
    }

    // apply sum/max to the final states__
    Tracking final_tracker;

    BOOST_FOREACH(Seq_States::value_type state, states){

      Tracking tracker = T[state];

      final_tracker.prob += tracker.prob;

      if(tracker.v_prob > final_tracker.v_prob) {
	final_tracker.v_path = tracker.v_path;
	final_tracker.v_prob = tracker.v_prob;
      }
    }

    std::cout << "Total probability of the observation sequence: " 
	      << final_tracker.prob << std::endl;
    std::cout << "Probability of the Viterbi path: " 
	      << final_tracker.v_prob << std::endl;
    std::cout << "The Viterbi path: " << std::endl;

    BOOST_FOREACH(Seq_States::value_type state, final_tracker.v_path){
      std::cout << "\tVState: " << state << std::endl;
    }
  }

}  // namespace Viterbi
