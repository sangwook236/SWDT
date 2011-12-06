#ifndef VITERBI_H
#define VITERBI_H 1

/*
  Implementation of http://en.wikipedia.org/wiki/Viterbi_algorithm
   It simplifies http://bozskyfilip.blogspot.com/2009/01/viterbi-algorithm-in-c-and-using-stl.html
   by using BOOST and cleaning-up the code
 */


#include <iostream>
#include <string>
#include <vector>
#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>


namespace Viterbi {

  // 
  // Define some useful types
  //  Here you can optimize mapping each state to an numeric id
  //  so that maps become vectors or matrices;
  //
  typedef double Probability;
  typedef std::string State;
  typedef std::vector<State> Seq_States;
  typedef boost::unordered_map<State,Probability> Map_State_Probability;
  typedef boost::unordered_map<State, Map_State_Probability> State_Map_State_Probability;
  
  //
  // computes total probability for observation
  // most likely viterbi path 
  // and probability of such path
  //
  void forward_viterbi(const Seq_States & obs, 
		       const Seq_States & states, 
		       Map_State_Probability & start_p, 
		       State_Map_State_Probability & trans_p, 
		       State_Map_State_Probability & emit_p);  
  //
  // Definition of an HMM
  //
  class HMM{
  
  public :
    HMM(){};
    
    // init hmm
    void init(void);
    
    friend std::ostream& operator << (std::ostream& os, HMM&);
        
    const Seq_States& get_states() const 
    { return states__; };
    const Seq_States& get_observations() const 
    { return observations__;};

    Map_State_Probability& get_start_probability() 
    { return start_probability__; } ;

    State_Map_State_Probability& get_transition_probability() 
    { return transition_probability__; }

    State_Map_State_Probability& get_emission_probability() 
    { return emission_probability__; }
      
  private:
    Seq_States states__;
    Seq_States observations__;
    Map_State_Probability start_probability__;
    State_Map_State_Probability transition_probability__,
				     emission_probability__;    
  };
}

#endif
