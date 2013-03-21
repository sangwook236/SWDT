#ifndef FSTSEARCHMONTECARLO_H
#define FSTSEARCHMONTECARLO_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    search_monte_carlo.hpp
   \brief   Repeatedly samples random subsets to eventually yield the one with highest criterion value
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at http://fst.utia.cz
   \date    March 2011
   \version 3.1.0.beta
   \note    FST3 was developed using gcc 4.3 and requires
   \note    \li Boost library (http://www.boost.org/, tested with versions 1.33.1 and 1.44),
   \note    \li (\e optionally) LibSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/, 
                tested with version 3.00)
   \note    Note that LibSVM is required for SVM related tools only,
            as demonstrated in demo12t.cpp, demo23.cpp, demo25t.cpp, demo32t.cpp, etc.

*/ /* 
=========================================================================
Copyright:
  * FST3 software (with exception of any externally linked libraries) 
    is copyrighted by Institute of Information Theory and Automation (UTIA), 
    Academy of Sciences of the Czech Republic.
  * FST3 source codes as presented here do not contain code of third parties. 
    FST3 may need linkage to external libraries to exploit its functionality
    in full. For details on obtaining and possible usage restrictions 
    of external libraries follow their original sources (referenced from
    FST3 documentation wherever applicable).
  * FST3 software is available free of charge for non-commercial use. 
    Please address all inquires concerning possible commercial use 
    of FST3, or if in doubt, to FST3 maintainer (see http://fst.utia.cz).
  * Derivative works based on FST3 are permitted as long as they remain
    non-commercial only.
  * Re-distribution of FST3 software is not allowed without explicit
    consent of the copyright holder.
Disclaimer of Warranty:
  * FST3 software is presented "as is", without warranty of any kind, 
    either expressed or implied, including, but not limited to, the implied 
    warranties of merchantability and fitness for a particular purpose. 
    The entire risk as to the quality and performance of the program 
    is with you. Should the program prove defective, you assume the cost 
    of all necessary servicing, repair or correction.
Limitation of Liability:
  * The copyright holder will in no event be liable to you for damages, 
    including any general, special, incidental or consequential damages 
    arising out of the use or inability to use the code (including but not 
    limited to loss of data or data being rendered inaccurate or losses 
    sustained by you or third parties or a failure of the program to operate 
    with any other programs).
========================================================================== */

#include <boost/smart_ptr.hpp>
#include <iostream>
#include <sstream>
#include <ctime>
#include "error.hpp"
#include "global.hpp"
#include "stopwatch.hpp"
#include "search.hpp"
#include "result_tracker.hpp"

/*============== Template parameter type naming conventions ==============
--------- Numeric types: -------------------------------------------------
DATATYPE - data sample values - usually real numbers (but may be integers
          in text processing etc.)
REALTYPE - must be real numbers - for representing intermediate results of 
          calculations like mean, covariance etc.
IDXTYPE - index values for enumeration of data samples - (nonnegative) integers, 
          extent depends on numbers of samples in data
DIMTYPE - index values for enumeration of features (dimensions), or classes (not 
          class sizes) - (nonnegative) integers, usually lower extent than IDXTYPE, 
          but be aware of expressions like _classes*_features*_features ! 
          in linearized representations of feature matrices for all classes
BINTYPE - feature selection marker type - represents ca. <10 different feature 
          states (selected, deselected, sel./desel. temporarily 1st nested loop, 2nd...)
RETURNTYPE - criterion value: real value, but may be extended in future to support 
          multiple values 
--------- Class types: ---------------------------------------------------
SUBSET       - class of class type Subset 
CLASSIFIER   - class implementing interface defined in abstract class Classifier 
EVALUATOR    - class implementing interface defined in abstract class Sequential_Step 
DISTANCE     - class implementing interface defined in abstract class Distance 
DATAACCESSOR - class implementing interface defined in abstract class Data_Accessor 
INTERVALCONTAINER - class of class type TIntervaller 
CONTAINER    - STL container of class type TInterval  
========================================================================== */

namespace FST {

/*! \brief Repeatedly samples random subsets to eventually yield the one with highest criterion value

	\note Requires user-set terminating condition, either in form of the number of random
	      trials accomplished or in form of maximum time spent or both.
	
	\note offers two ways of random subset generation:
	      a) set_cardinality_randomization(const DIMTYPE d_from, const DIMTYPE d_to) invokes
	         the first option, i.e., when generating a random subset, the subset size
	         is first chosen randomly from [d_from,d_to], then features are randomly
	         selected until the required number is reached
	      b) set_cardinality_randomization(const float d_prob) invokes
	         the second option, where each feature is randomly included/excluded
	         with probability d_prob. The expected subset size is thus d_prob*no_of_all_features,
	         but can be actually anything from [0,no_of_all_features].
*/

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search_Monte_Carlo : public Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent;
	typedef boost::shared_ptr<CRITERION> PCriterion;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Search_Monte_Carlo(const unsigned long trials_limit=1000, const unsigned long time_limit=0, const unsigned int time_limit_check_freq=1):Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>() {notify("Search_Monte_Carlo constructor."); set_stopping_condition(trials_limit,time_limit,time_limit_check_freq); reset_cardinality_randomization();}
	virtual ~Search_Monte_Carlo() {notify("Search_Monte_Carlo destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout); //!< returns found subset of target_d features (optimizes cardinality if target_d==0) + criterion value

	//! set maximum number of random trials or maximum time in seconds or both
	void set_stopping_condition(const unsigned long trials_limit=1000, const unsigned long time_limit=0, const unsigned int time_limit_check_freq=100) {if(trials_limit==0 && time_limit==0) {_trials_limit=1000; _time_limit=0;} else {_trials_limit=trials_limit; _time_limit=time_limit;} if(time_limit_check_freq==0) _time_limit_check_freq=100; else _time_limit_check_freq=time_limit_check_freq; }

	// set the way of random subset (size) generation
	void set_cardinality_randomization(const DIMTYPE d_from, const DIMTYPE d_to) {if(d_from>0 && d_from<=d_to) {_d_from=d_from; _d_to=d_to;} else {_d_from=_d_to=0;}} //!< generate subset size first from [d_from,d_to], then randomly choose exactly that number of features
	void set_cardinality_randomization(const float d_prob) {if(d_prob>0.0 && d_prob<1.0) _d_prob=d_prob; else _d_prob=0.0;} //!< for each feature roll the dice and select it with probability d_prob
	void reset_cardinality_randomization() {_d_from=_d_to=0; _d_prob=0.0;} //!< equivalent to set_cardinality_randomization(1,no_of_all_features)

	virtual std::ostream& print(std::ostream& os) const;
		
protected:
	unsigned long _time_limit; //!< max time allowed to be spent in seconds
	unsigned long _trials_limit; //!< max no of candidates evaluated
	unsigned int _time_limit_check_freq; //!< number of trials before next time check
	DIMTYPE _d_from, _d_to; //!< generate random subsets of size within these limits
	float _d_prob; //!< active d_prob must be from (0,1]; generates random subset of random size where the mean size over a series of calls is d_prob*no_of_all_features
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
std::ostream& Search_Monte_Carlo<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::print(std::ostream& os) const 
{
	os << "Random search  [Search_Monte_Carlo("; 
	if(_trials_limit>0) os << "maxtrials="<<_trials_limit<<", "; 
	if(_time_limit>0) os<<"seconds="<<_time_limit<<", "; 
	if(_d_prob>0.0) os<<"d_prob="<<_d_prob; else os<<"d_from="<<_d_from<<", d_to="<<_d_to; os<<")"; 
	if(parent::result_tracker_active()) os << " with " << *parent::_tracker; os<<"]"; 
	return os;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Search_Monte_Carlo<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	StopWatch swatch;
	assert(sub);
	assert(crit);
	const DIMTYPE n=sub->get_n();
	assert(target_d>=0 && target_d<n);
	PSubset tmp_sub(new SUBSET(n));
	bool initialized=false;

	bool track=parent::result_tracker_active(); if(track) parent::_tracker->set_output_detail(parent::get_output_detail());

	if(parent::output_normal()) {
		std::ostringstream sos; 
		sos << "---------------------------------------" << std::endl;
		sos << "Starting " << *this << std::endl;
		sos << "with Criterion: " << *crit << std::endl;
		if(target_d==0) sos << "Subset size to be optimized." << std::endl; else sos << "Target subset size set to: " << target_d << std::endl;
		sos << std::endl << std::flush;
		syncout::print(os,sos);
	}
	assert(_trials_limit!=0 || _time_limit!=0);
	assert(_time_limit_check_freq>0);
	unsigned long _trials=0;
	unsigned long _time=0;
	DIMTYPE _d_from_tmp=_d_from, _d_to_tmp=_d_to;
	if(_d_to_tmp<1 || _d_to_tmp>n) _d_to_tmp=n;
	if(_d_from_tmp<1 || _d_from_tmp>_d_to_tmp) _d_from_tmp=_d_to_tmp;
	assert(_d_from_tmp>=1 && _d_from_tmp<=_d_to_tmp && _d_to_tmp<=n);
	
	RETURNTYPE val;
	while( (_trials_limit==0 || _trials<_trials_limit) && (_time_limit==0 || _time<_time_limit) )
	{
		if(target_d==0) {
			if(_d_prob>0.0) do {tmp_sub->make_random_subset(_d_prob);} while(tmp_sub->get_d()==0);
			else tmp_sub->make_random_subset(_d_from_tmp,_d_to_tmp);
		} else tmp_sub->make_random_subset(target_d);

		if(!crit->evaluate(val,tmp_sub)) return false;
		if(track) parent::_tracker->add(val,tmp_sub);
		if(!initialized || val>result || (val==result && tmp_sub->get_d()<sub->get_d())) {
			result=val;
			sub->stateless_copy(*tmp_sub);
			initialized=true;
			if(parent::output_normal()) {std::ostringstream sos; sos << "new MAXCRIT="<<val<< ", trial=" << _trials << ", " << *tmp_sub << std::endl << swatch << std::endl << std::endl << std::flush; syncout::print(os,sos);}
		}
		_trials++;
		if(_trials%_time_limit_check_freq==0) _time=(unsigned long)swatch.time_elapsed();
	}
	assert(initialized);
	if(parent::output_normal()) {
		std::ostringstream sos; 
		sos << "......................................." << std::endl;
		sos << "Search_Monte_Carlo() search finished after "<<_trials<<" trials. " << swatch << std::endl;
		sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
		syncout::print(os,sos);
	}
	return true;
}

} // namespace
#endif // FSTSEARCHMONTECARLO_H ///:~
