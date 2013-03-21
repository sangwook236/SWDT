#ifndef FSTSEARCHSEQSTEPSTRAIGHTTHREADED_H
#define FSTSEARCHSEQSTEPSTRAIGHTTHREADED_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    seq_step_straight_threaded.hpp
   \brief   Implements threaded version of sequential selection step in sequential search type of methods 
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

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/function.hpp>
#include <boost/static_assert.hpp>
#include <boost/bind.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>
#include <sstream>
#include "error.hpp"
#include "global.hpp"
#include "seq_step.hpp"
#include "candidate_evaluator_threaded.hpp"

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

/*! \brief Implements threaded version of sequential selection step in sequential search type of methods  

 Concurrently evaluates each subset candidate using clones of the supplied criterion and returns the
 one subset that yielded highest criterion value 
 
 \warning Note that Threaded_Criterion_Evaluator keeps the pointer to criterion
 object as supplied to evaluate_candidates. At the time of first launch of the
 worker threads the pointed crit is cloned to get as many clones as there will be threads.
 It is important to realize that the clones are kept and reused as long as the 'crit' 
 parameter to evaluate_candidates remains the same. Note that the state of the clones is 
 not! synchronized with the possibly changing state of the pointed template criterion.
 This has implications especially when used with Criterion_Wrapper, which depends
 on current data split. If the current split changes, the change does not propagate
 to clones and the results are thus obtained for the wrong part of data. Therefore,
 to ensure the correct results are achieved, better re-allocate the Sequential_Step_Straight_Threaded
 object so that it's criterion clones are re-generated whenever the current context of
 the primary criterion changes - typically after every data split change.
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads=2>
class Sequential_Step_Straight_Threaded : public Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent;
	typedef boost::shared_ptr<CRITERION> PCriterion;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Sequential_Step_Straight_Threaded():Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>() 
		{BOOST_STATIC_ASSERT(max_threads>0); notify("Sequential_Step_Straight_Threaded constructor.");}
	virtual ~Sequential_Step_Straight_Threaded() {notify("Sequential_Step_Straight_Threaded destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	inline bool evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level=1, std::ostream& os=std::cout); //!< chooses among subsets offered by sub->get*CandidateSubset()

	virtual std::ostream& print(std::ostream& os) const {os << "Sequential_Step_Threaded(threads="<<max_threads<<")"; if(parent::result_tracker_active()) os << " with " << *parent::_tracker; return os;}
protected:
	Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads> _evaluator;
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
bool Sequential_Step_Straight_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level, std::ostream& os)
{
	assert(sub);
	assert(crit);
	assert(_generalization_level>=1 && _generalization_level<=sub->get_n());
	_evaluator.initialize(crit, sub->getNoOfCandidateSubsets(_generalization_level,false/*reverse*/));
	bool b=sub->getFirstCandidateSubset(_generalization_level,false/*reverse*/);
	if(!b) return false; // no candidate subsets can be traversed for whatever reason
	_evaluator.add_candidate(sub);		
	b=sub->getNextCandidateSubset(); // evaluate the remaining candidates
	if(parent::output_detailed()) {std::ostringstream sos; sos << "Step...: sub=" << *sub << std::endl << std::endl << std::flush; syncout::print(os,sos);}
	while(b) {
		_evaluator.add_candidate(sub);
		b=sub->getNextCandidateSubset();
		if(parent::output_detailed()) {std::ostringstream sos; sos << "Step...: sub=" << *sub << std::endl << std::endl << std::flush; syncout::print(os,sos);}
	}
	_evaluator.results_to_tracker(parent::_tracker);
	PSubset psub;
	DIMTYPE dummyfeature;
	if(!_evaluator.getBestResult(result,psub,dummyfeature)) return false;
	assert(psub);
	assert(psub->get_n()==sub->get_n());
	sub->stateless_copy(*psub);
	return true;
}

} // namespace
#endif // FSTSEARCHSEQSTEPSTRAIGHTTHREADED_H ///:~
