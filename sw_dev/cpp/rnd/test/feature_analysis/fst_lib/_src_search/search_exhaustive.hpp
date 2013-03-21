#ifndef FSTSEARCHEXHAUSTIVE_H
#define FSTSEARCHEXHAUSTIVE_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    search_exhaustive.hpp
   \brief   Defines interface for search method implementations
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

/*! \brief Implements exhaustive (optimal) search yielding optimal feature subset with respect to chosen criterion

	\note Due to possibly high number of subsets to be tested expect
	excessive computational time. In case of result tracking the excessive
	number of possible combinations may consume unacceptable amount of memory - to prevent
	this it is highly recommended to set up result tracker storage limit.
*/

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search_Exhaustive : public Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent;
	typedef boost::shared_ptr<CRITERION> PCriterion;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Search_Exhaustive():Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>() {notify("Search_Exhaustive constructor.");}
	virtual ~Search_Exhaustive() {notify("Search_Exhaustive destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout); //!< returns found subset of target_d features (optimizes cardinality if target_d==0) + criterion value

	virtual std::ostream& print(std::ostream& os) const {os << "Exhaustive search  [Search_Exhaustive()"; if(parent::result_tracker_active()) os << " with " << *parent::_tracker; os<<"]"; return os;};
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Search_Exhaustive<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	StopWatch swatch;
	notify("Search_Exhaustive::search().");
	assert(sub);
	assert(crit);
	const DIMTYPE n=sub->get_n();
	assert(target_d>=0 && target_d<n);
	boost::scoped_ptr<SUBSET> tmp_step_sub; // (lazy allocation)
	tmp_step_sub.reset(new SUBSET(n));

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

	RETURNTYPE val,bestval;
	DIMTYPE t, t_from=1, t_to=n;
	if(target_d>0) {t_from=target_d; t_to=target_d;}
	bool initialized=false;
	for(t=t_from;t<=t_to;t++)
	{
		sub->deselect_all();
		bool b=sub->getFirstCandidateSubset(t,false/*reverse*/);
		if(!b) return false; // no candidate subsets can be traversed for whatever reason	
		if(!crit->evaluate(val,sub)) return false; // store first candidate as the initially best
		if(track) parent::_tracker->add(val,sub);
		if(!initialized || val>bestval) {
			bestval=val;
			tmp_step_sub->stateless_copy(*sub);
			initialized=true;
			if(parent::output_normal()) {std::ostringstream sos; sos << "new MAXCRIT="<<bestval<<", " << *tmp_step_sub << std::endl << swatch << std::endl << std::endl << std::flush; syncout::print(os,sos);}
		}
	
		b=sub->getNextCandidateSubset(); // evaluate the remaining candidates
		while(b) {
			if(!crit->evaluate(val,sub)) return false; 
			if(track) parent::_tracker->add(val,sub);
			if(val>bestval) {
				bestval=val;
				tmp_step_sub->stateless_copy(*sub);
				if(parent::output_normal()) {std::ostringstream sos; sos << "new MAXCRIT="<<bestval<<", " << *tmp_step_sub << std::endl << swatch << std::endl << std::endl << std::flush; syncout::print(os,sos);}
			}
			b=sub->getNextCandidateSubset();
		}
	}
	assert(initialized);
	sub->stateless_copy(*tmp_step_sub);
	result=bestval;
	if(parent::output_normal()) {
		std::ostringstream sos; 
		sos << "......................................." << std::endl;
		sos << "Search_Exhaustive() search finished. " << swatch << std::endl;
		sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
		syncout::print(os,sos);
	}
	return true;
}

} // namespace
#endif // FSTSEARCHEXHAUSTIVE_H ///:~
