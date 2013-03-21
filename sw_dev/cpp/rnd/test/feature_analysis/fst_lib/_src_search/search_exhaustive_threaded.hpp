#ifndef FSTSEARCHEXHAUSTIVETHREADED_H
#define FSTSEARCHEXHAUSTIVETHREADED_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    search_exhaustive_threaded.hpp
   \brief   Implements threaded version of exhaustive (optimal) search yielding optimal feature subset with respect to chosen criterion 
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
#include "thread_pool.hpp"
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
EVALUATOR    - class implementing interface defined in abstract class Search 
DISTANCE     - class implementing interface defined in abstract class Distance 
DATAACCESSOR - class implementing interface defined in abstract class Data_Accessor 
INTERVALCONTAINER - class of class type TIntervaller 
CONTAINER    - STL container of class type TInterval  
========================================================================== */

namespace FST {

/*! \brief Implements threaded version of exhaustive (optimal) search yielding optimal feature subset with respect to chosen criterion

 Concurrently evaluates each subset candidate using clones of the supplied criterion and returns the
 one subset that yielded highest criterion value 
 
 \note Due to possibly high number of subsets to be tested expect
 excessive computational time. In case of result tracking the excessive
 number of possible combinations may consume unacceptable amount of memory - to prevent
 this it is highly recommended to set ResultTracker storage limit.
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads=2>
class Search_Exhaustive_Threaded : public Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent;
	typedef boost::shared_ptr<CRITERION> PCriterion;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Search_Exhaustive_Threaded():Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>() 
		{BOOST_STATIC_ASSERT(max_threads>0); notify("Search_Exhaustive_Threaded constructor.");}
	virtual ~Search_Exhaustive_Threaded() {notify("Search_Exhaustive_Threaded destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout); //!< returns found subset of target_d features (optimizes cardinality if target_d==0) + criterion value

	virtual std::ostream& print(std::ostream& os) const {os << "Exhaustive search  [Search_Exhaustive_Threaded(threads="<<max_threads<<")"; if(parent::result_tracker_active()) os << " with " << *parent::_tracker; os<<"]"; return os;}
protected:
	PCriterion template_crit; //!< to detect change of criterion inbetween search() calls
	//! thread-local storage of current subset candidate, criterion clone, and tracker clone
	typedef struct {
		PCriterion crit;
		typename parent::PResultTracker tracker;
		PSubset bestsub;
		RETURNTYPE bestval;
		bool bestval_available;
		std::ostream* os;
	} ThreadLocal;
	ThreadLocal tlocal[max_threads];
	
	void evaluator_thread(unsigned int idx);

	bool get_candidate(PSubset &sub); //!< generates next candidate, then copies it to sub	
	boost::mutex mutex_candidate_generator;
	PSubset _candidate_generator;
	DIMTYPE _cardinality, _target_d;
	bool _generate_first; //!< for current cardinality determines whether to call getFirstCandidateSubset or getNextCandidateSubset

	boost::shared_ptr<StopWatch> swatch;
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
void Search_Exhaustive_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::evaluator_thread(unsigned int idx)
{
	assert(idx>=0 && idx<max_threads);
	assert(tlocal[idx].bestsub);
	assert(tlocal[idx].bestsub->get_n()>0);
	assert(tlocal[idx].crit);
	assert(swatch);
	ThreadLocal &tl=tlocal[idx];
	std::ostream& os=*(tlocal[idx].os);

	PSubset sub(new SUBSET(tl.bestsub->get_n()));
	RETURNTYPE result;	

	if(parent::output_detailed()) {std::ostringstream sos; sos << "Thread " << idx << " started." << std::endl << std::endl << std::flush; syncout::print(os,sos);}
	while(get_candidate(sub))
	{
		if(!tl.crit->evaluate(result,sub)) throw fst_error("Search_Exhaustive_Threaded::evaluate_candidate() criterion evaluation failure.");
		
		if(tl.tracker) tl.tracker->add(result,sub);

		if(!tl.bestval_available || result>tl.bestval) {// not needed here due to candidate generating order: || (result==tl.bestval && sub->get_d()<tl.bestsub->get_d())) {
			tl.bestval=result;
			tl.bestsub->stateless_copy(*sub);
			tl.bestval_available=true;
			if(parent::output_normal()) {std::ostringstream sos; sos << "THREAD " << idx << " new MAXCRIT="<<result<<", " << *sub << std::endl << *swatch << std::endl << std::endl << std::flush; syncout::print(os,sos);}
		}
	}
	if(parent::output_detailed()) {std::ostringstream sos; sos << "Thread " << idx << " finished." << std::endl << std::endl << std::flush; syncout::print(os,sos);}
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
bool Search_Exhaustive_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::get_candidate(PSubset &sub)
{
	boost::mutex::scoped_lock lock(mutex_candidate_generator);
	assert(sub);
	assert(sub->get_n()==_candidate_generator->get_n());
	assert(_cardinality>0 && _cardinality<=_candidate_generator->get_n());
	
	restart: if(_generate_first) {
		_candidate_generator->deselect_all();
		bool b=_candidate_generator->getFirstCandidateSubset(_cardinality,false/*reverse*/);
		if(!b) return false; // no candidate subsets can be traversed for whatever reason
		sub->stateless_copy(*_candidate_generator);
		_generate_first=false;
	} else {
		bool b=_candidate_generator->getNextCandidateSubset(); // evaluate the remaining candidates
		if(b) sub->stateless_copy(*_candidate_generator);
		else {
			if(_target_d>0 || ++_cardinality>_candidate_generator->get_n()) return false; // all configurations have been already generated
			_generate_first=true;
			goto restart;
		}
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
bool Search_Exhaustive_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	swatch.reset(new StopWatch);
	notify("Search_Exhaustive_Threaded::search().");
	assert(sub);
	assert(crit);
	const DIMTYPE n=sub->get_n();
	assert(target_d>=0 && target_d<=n);

	if(parent::result_tracker_active()) parent::_tracker->set_output_detail(parent::get_output_detail());

	if(parent::output_normal()) {
		std::ostringstream sos; 
		sos << "---------------------------------------" << std::endl;
		sos << "Starting " << *this << std::endl;
		sos << "with Criterion: " << *crit << std::endl;
		if(target_d==0) sos << "Subset size to be optimized." << std::endl; else sos << "Target subset size set to: " << target_d << std::endl;
		sos << std::endl << std::flush;
		syncout::print(os,sos);
	}
	
	// test master crit pointer change ? then recreate criterion clones
	if(template_crit!=crit) {
		for(unsigned int i=0;i<max_threads;i++) tlocal[i].crit.reset(crit->clone());
		template_crit=crit; 
	}

	// initialize all other thread local structures
	for(unsigned int i=0;i<max_threads;i++) {
		if(!tlocal[i].bestsub || tlocal[i].bestsub->get_n()!=n) tlocal[i].bestsub.reset(new SUBSET(n));
		if(parent::_tracker) tlocal[i].tracker.reset(parent::_tracker->stateless_clone());
		tlocal[i].bestval_available=false;
		tlocal[i].os=&os;
	}
	
	// status indiactors to initialize get_candidate() operation
	_candidate_generator.reset(new SUBSET(n));
	_target_d=target_d;
	if(_target_d>0) _cardinality=_target_d; else _cardinality=1;
	_generate_first=true;
	
	// launch worker threads and wait for them to finish
	ThreadPool<max_threads> tp;
	for(unsigned int i=0;i<max_threads;i++) tp.go(boost::bind(&Search_Exhaustive_Threaded::evaluator_thread, this, i));
	tp.join_all();

	if(parent::output_normal()) {std::ostringstream sos; sos << *swatch << std::endl << std::endl << std::flush; syncout::print(os,sos);}
	// join thread-local trackers' contents
	if(parent::_tracker) {
		for(unsigned int i=0;i<max_threads;i++) {
			if(parent::output_detailed()) {std::ostringstream sos; sos << "Joining thread ["<<i<<"] tracker.. (size=" << tlocal[i].tracker->size() << ")" << std::endl << std::flush; syncout::print(os,sos);}
			parent::_tracker->join(*(tlocal[i].tracker.get()));
			tlocal[i].tracker->clear();
		}
		if(parent::output_detailed()) {std::ostringstream sos; sos << "Joined thread local trackers (size=" << parent::_tracker->size() << ")." << std::endl << *swatch << std::endl << std::endl << std::flush; syncout::print(os,sos);}
	}
	
	// identify the globally best result
	unsigned int bestidx=0;
	for(unsigned int i=1;i<max_threads;i++) if(tlocal[i].bestval>tlocal[bestidx].bestval) bestidx=i;
	sub->stateless_copy(*tlocal[bestidx].bestsub);
	result=tlocal[bestidx].bestval;
	if(parent::output_normal()) {
		std::ostringstream sos; 
		sos << "......................................." << std::endl;
		sos << "Search_Exhaustive_Threaded() search finished. " << *swatch << std::endl;
		sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
		syncout::print(os,sos);
	}
	swatch.reset();
	return true;
}

} // namespace
#endif // FSTSEARCHEXHAUSTIVETHREADED_H ///:~
