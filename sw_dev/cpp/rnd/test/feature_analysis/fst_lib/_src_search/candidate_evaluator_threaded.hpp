#ifndef FSTCANDIDATEEVALUATORTHREADED_H
#define FSTCANDIDATEEVALUATORTHREADED_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    candidate_evaluator_threaded.hpp
   \brief   Implements concurrent evaluation of a criterion on a set of subset candidates
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
#include "error.hpp"
#include "global.hpp"
#include "thread_pool.hpp"
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

enum CriterionStatus {
  CRIT_NONEXISTENT=0,
  CRIT_READY=1,
  CRIT_ENGAGED=2
};

/*! \brief Implements concurrent evaluation of a criterion on a set of subset candidates

 Concurrently evaluates each subset candidate using clones of the supplied criterion and
 at most max_threads worker threads.
 Returns the one subset that yiels highest criterion value, or iterates through all obtained results
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads=2>
class Candidate_Evaluator_Threaded {
public:
	typedef boost::shared_ptr<CRITERION> PCriterion;
	typedef boost::shared_ptr<SUBSET> PSubset;
	typedef boost::shared_ptr<Result_Tracker<RETURNTYPE,SUBSET> > PResultTracker;

	Candidate_Evaluator_Threaded() : initialized(false), working(false), _processed_candidates(0), _available_candidates(0), stop_all_threads(false), _reading_result(0) {
		BOOST_STATIC_ASSERT(max_threads>0);
		for(unsigned int i=0;i<max_threads;i++) tlocal[i].status=CRIT_NONEXISTENT; // not available, not allocated
		notify("Candidate_Evaluator_Threaded constructor.");
	}
	virtual ~Candidate_Evaluator_Threaded() {notify("Candidate_Evaluator_Threaded destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	void initialize(const PCriterion &crit, const DIMTYPE max_candidates); //!< to prevent race conditions due to vector resizing max_candidates must be set greater or equal than the maximum number of candidates that will be added
	bool add_candidate(const PSubset &sub, const DIMTYPE feature=0); //!< adds candidate subset to the evaluation pool

	// any of the following readers first waits the threads to finish the complete job. Further calls to add_candidate will be treated as part of a new job
	bool getBestResult(RETURNTYPE &result, PSubset &sub, DIMTYPE &feature); //!< returns pointer to (does not copy sub contents) whichever candidate yielded the highest criterion value (first waits for all threads to finish)
	bool getFirstResult(RETURNTYPE &result, PSubset &sub, DIMTYPE &feature); //!< iterator, returns pointer (does not copy sub contents)
	bool getNextResult(RETURNTYPE &result, PSubset &sub, DIMTYPE &feature); //!< iterator
	
	bool results_to_tracker(const PResultTracker tracker);

	virtual std::ostream& print(std::ostream& os) const {os << "Sequential_Step_Threaded(max_threads="<<max_threads<<")"; return os;}
protected:
	PCriterion template_crit; //!< to detect change of employed criterion inbetween search() calls
	bool initialized;
	bool working;

	//! thread-local storage of current subset candidate, criterion clone, and tracker clone
	typedef struct {
		PCriterion crit;
		CriterionStatus status; // CRIT_NONEXISTENT or CRIT_READY or CRIT_ENGAGED
	} ThreadLocal;
	ThreadLocal tlocal[max_threads];
	
	ThreadPool<max_threads> tp;
	void evaluator_thread(unsigned int tidx);
	bool get_candidate(DIMTYPE &idx); //!< points the calling thread to next candidate to be processed
	         DIMTYPE _processed_candidates;
	volatile DIMTYPE _available_candidates; //!< to sychronize get_candidate() with adding new candidates to cres[]
	
	boost::mutex mutex_candidate_getter; //!< to sychronize threads when requesting new cres[] candidate through get_candidate()
	boost::condition condition_available; //!< to announce availability of new candidate for processing
	volatile bool stop_all_threads;

	//! temporary storage of subset candidate evaluation results
	typedef struct {
		RETURNTYPE result;
		PSubset sub;
		DIMTYPE feature; // optional
	} CandidateResult;
	vector<CandidateResult> cres;

	DIMTYPE _reading_result; //!< iterating index for get*Result readers
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
void Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::initialize(const PCriterion &crit, const DIMTYPE max_candidates) 
{ // max_candidates is optional, if nonzero, it reserves enough storage space to later save time doing that
	assert(crit);
	assert(max_candidates>0);
	
	if(working) {
		stop_all_threads=true;
		condition_available.notify_all();
		tp.join_all(); 
	}
	_processed_candidates=0;
	_available_candidates=0;
	_reading_result=0;
	stop_all_threads=false;
	working=true;

	if(!initialized || crit!=template_crit) {// master crit pointer change ? update criterion clones in clone array ?
		template_crit=crit;
		for(unsigned int i=0;i<max_threads;i++) {tlocal[i].crit.reset(); tlocal[i].status=CRIT_NONEXISTENT;} // 0=not available, not allocated
	}
	if((DIMTYPE)(cres.size())<max_candidates) cres.resize(max_candidates);

	initialized=true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
void Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::evaluator_thread(unsigned int tidx)
{
	assert(tidx>=0 && tidx<max_threads);
	ThreadLocal &tl=tlocal[tidx];
	assert(tl.crit);

	DIMTYPE idx;

	while(get_candidate(idx))
	{
		CandidateResult &cr=cres[idx];
		if(!tl.crit->evaluate(cr.result,cr.sub)) throw fst_error("Candidate_Evaluator_Threaded::crit->evaluate(cr.result,cr.sub) criterion evaluation failure.");
	}
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
bool Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::get_candidate(DIMTYPE &idx)
{
	boost::mutex::scoped_lock lock(mutex_candidate_getter);
	while(_processed_candidates==_available_candidates && !stop_all_threads) condition_available.wait(lock);
	if(_processed_candidates==_available_candidates && stop_all_threads) return false;
	assert(_processed_candidates<_available_candidates);
	idx=_processed_candidates++;
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
bool Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::add_candidate(const PSubset &sub, const DIMTYPE feature) 
{
	assert(initialized);
	assert(sub);
	assert(sub->get_d()>0);
	
	if(!working) {//start with new group of candidates
		_processed_candidates=0;
		_available_candidates=0;
		stop_all_threads=false;
		working=true;
	}	
	
	if(_available_candidates>=cres.size()) return false;

	if(!cres[_available_candidates].sub || cres[_available_candidates].sub->get_n()!=sub->get_n()) cres[_available_candidates].sub.reset(new SUBSET(sub->get_n()));
	cres[_available_candidates].sub->stateless_copy(*sub);
	cres[_available_candidates].feature=feature;
	
	if(_available_candidates<max_threads) {
		if(tlocal[_available_candidates].status==CRIT_NONEXISTENT || tlocal[_available_candidates].crit!=template_crit) {tlocal[_available_candidates].crit.reset(template_crit->clone()); tlocal[_available_candidates].status=CRIT_READY;}
		tp.go(boost::bind(&Candidate_Evaluator_Threaded::evaluator_thread, this, _available_candidates));
	}

	++_available_candidates;
	condition_available.notify_one();
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
bool Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::getBestResult(RETURNTYPE &result, PSubset &sub, DIMTYPE &feature) 
{// returns whichever candidate yielded the highest criterion value (waits for all threads to finish)
	if(working) {
		stop_all_threads=true;
		condition_available.notify_all();
		tp.join_all(); 
		working=false;
	}	
	DIMTYPE bestidx=0;
	for(DIMTYPE i=1;i<_available_candidates;i++) if(cres[i].result>cres[bestidx].result) bestidx=i;
	sub=cres[bestidx].sub;
	result=cres[bestidx].result;
	feature=cres[bestidx].feature;
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
bool Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::getFirstResult(RETURNTYPE &result, PSubset &sub, DIMTYPE &feature) 
{// returns whichever candidate yielded the highest criterion value (waits for all threads to finish)
	if(_available_candidates==0 || cres.size()==0) return false;
	if(working) {
		stop_all_threads=true;
		condition_available.notify_all();
		tp.join_all(); 
		working=false;
	}	
	_reading_result=0;
	sub=cres[_reading_result].sub;
	result=cres[_reading_result].result;
	feature=cres[_reading_result].feature;
	_reading_result=1;
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
bool Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::getNextResult(RETURNTYPE &result, PSubset &sub, DIMTYPE &feature) 
{// returns whichever candidate yielded the highest criterion value (waits for all threads to finish)
	if(working || _reading_result>=_available_candidates) return false;
	assert(_reading_result<cres.size());
	sub=cres[_reading_result].sub;
	result=cres[_reading_result].result;
	feature=cres[_reading_result].feature;
	++_reading_result;
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
bool Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::results_to_tracker(const PResultTracker tracker)
{
	if(!tracker || _available_candidates==0) return false;
	if(working) {
		stop_all_threads=true;
		condition_available.notify_all();
		tp.join_all(); 
		working=false;
	}	
	for(DIMTYPE i=0;i<_available_candidates;i++) tracker->add(cres[i].result,cres[i].sub);
	return true;
}

} // namespace
#endif // FSTCANDIDATEEVALUATORTHREADED_H ///:~
