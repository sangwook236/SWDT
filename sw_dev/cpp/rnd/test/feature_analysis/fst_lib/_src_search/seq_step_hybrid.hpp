#ifndef FSTSEARCHSEQSTEPHYBRID_H
#define FSTSEARCHSEQSTEPHYBRID_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    seq_step_hybrid.hpp
   \brief   Implements hybrid selection step in sequential search type of methods 
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
#include <list>
#include "error.hpp"
#include "global.hpp"
#include "seq_step.hpp"

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

// NOTE: adds the option of "hybrid" Step()s, i.e., pre-filtering of candidates using a second - presumably faster - criterion, to save time when evaluating the main criterion

	// NOTE: when enabled, the following implies "hybrid" search where all candidates are first
	//       evaluated using filtercrit, then only keep_perc% is passed to
	//       choose from using the main criterion
	// NOTE: keep_perc==0 bypasses Step()::crit and returns the best subset only with respect to filtercrit
	//       keep_perc==100 bypasses the pre-filtering mechanism and returns standard Step() output

//! Implements hybrid selection step in sequential search type of methods 
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class FILTERCRITERION>
class Sequential_Step_Hybrid : public Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent;
	typedef boost::shared_ptr<FILTERCRITERION> PFilterCriterion;
	typedef boost::shared_ptr<CRITERION> PCriterion;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Sequential_Step_Hybrid(const PFilterCriterion filtercrit, const DIMTYPE keep_perc, const DIMTYPE keep_max=1000):Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(),track(false) {assert(0<=keep_perc && keep_perc<=100); assert(keep_max>=0); assert(filtercrit); _filtercrit=filtercrit; _keep_perc=keep_perc; _keep_max=keep_max; _listsize=0; notify("Sequential_Step_Hybrid constructor.");}
	virtual ~Sequential_Step_Hybrid() {notify("Sequential_Step_Hybrid destructor.");}

	void set_prefiltering_limits(const DIMTYPE keep_perc, const DIMTYPE keep_max) {assert(0<=keep_perc && keep_perc<=100); assert(keep_max>=0); _keep_perc=keep_perc; _keep_max=keep_max;}
	DIMTYPE get_prefiltering_perc() const {return _keep_perc;}
	DIMTYPE get_prefiltering_max() const {return _keep_max;}

	// NOTE: crit is expected to be already initialize()d before use here
	inline bool evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level=1, std::ostream& os=std::cout); //!< chooses among subsets offered by sub->get*CandidateSubset()

	virtual std::ostream& print(std::ostream& os) const;
protected:
	inline bool filter_candidates(const PSubset sub, const DIMTYPE _generalization_level, std::ostream& os);
	inline bool test_candidate(const PSubset sub);
	
	PFilterCriterion _filtercrit;
	DIMTYPE _keep_perc; //!< filter out (100-_keep_perc) % of candidate subsets
	DIMTYPE _keep_max; //!< when filtering keep maximally _keep_max candidate subsets (restriction on CANDIDATELIST size)
	
	//! Nested class to hold [Subset,criterion value] pair in the course of hybrid feature candidate evaluation in Sequential_Step_Hybrid
	class SubsetCandidate {
	public:
		SubsetCandidate(const RETURNTYPE value, const PSubset sub) : _value(value), _sub(new SUBSET(*sub)) {}
		const RETURNTYPE _value;
		const PSubset _sub;
	};
	typedef std::list<SubsetCandidate> CANDIDATELIST;

	DIMTYPE _listsize;
	CANDIDATELIST _filterlist; // (lazy allocation)
	typename CANDIDATELIST::iterator iter; 

private:
	boost::scoped_ptr<SUBSET> tmp_step_sub; // (lazy allocation)
	bool track;
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class FILTERCRITERION>
std::ostream& Sequential_Step_Hybrid<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,FILTERCRITERION>::print(std::ostream& os) const 
{
	os << "Sequential_Step_Hybrid(keep_perc="<<_keep_perc<<", max="<<_keep_max<<")"; 
	if(_filtercrit) os << " with filter " << *_filtercrit;
	if(parent::result_tracker_active()) os << " with " << *parent::_tracker; 
	return os;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class FILTERCRITERION>
bool Sequential_Step_Hybrid<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,FILTERCRITERION>::test_candidate(const PSubset sub)
{
	assert(sub);
	assert(_filtercrit);
	RETURNTYPE val;
	DIMTYPE _idx=0;
	assert(sub->get_d()!=0);
	if(!_filtercrit->evaluate(val,sub)) return false; //if(track) parent::_tracker->add(val,sub); // does not make sense to mix results of various criteria
	for(iter=_filterlist.begin(); _idx<_listsize && iter!=_filterlist.end() && (val<iter->_value || (val==iter->_value && sub->get_d()>=iter->_sub->get_d())); iter++) {++_idx;}
	if(_idx<_listsize) { // candidate good enough to be inserted
		_filterlist.insert(iter,SubsetCandidate(val,sub));
		if(_filterlist.size()>_listsize) _filterlist.pop_back();
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class FILTERCRITERION>
bool Sequential_Step_Hybrid<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,FILTERCRITERION>::filter_candidates(const PSubset sub, const DIMTYPE _generalization_level, std::ostream& os)
{
	assert(sub);
	_filterlist.clear();
	assert(0<=_keep_perc && _keep_perc<=100);
	assert(_keep_max>0);
	assert(_generalization_level>=1 && _generalization_level<=sub->get_n());
	
	// determine maximum _filterlist size
	_listsize=DIMTYPE(((double)_keep_perc/100.0)*sub->getNoOfCandidateSubsets(_generalization_level,false/*reverse*/));
	if(_keep_max<_listsize) _listsize=_keep_max;
	if(_listsize==0) _listsize=1;
	
	// first build candidate list _filterlist (pre-filter using _filtercrit criterion)
	bool b=sub->getFirstCandidateSubset(_generalization_level,false/*reverse*/); if(!b) return false; // no candidate subsets can be traversed for whatever reason
	if(!test_candidate(sub)) return false;
	b=sub->getNextCandidateSubset(); // evaluate the remaining candidates
	while(b) {
		if(!test_candidate(sub)) return false;
		b=sub->getNextCandidateSubset();
	}
	assert(_filterlist.size()>0);
	if(parent::output_detailed()) {
		std::ostringstream sos; 
		sos << "candidate list of "<<_filterlist.size() << ":" << std::endl;
		for(iter=_filterlist.begin(); iter!=_filterlist.end(); iter++) {sos << iter->_value << " ~ " << *(iter->_sub) << std::endl;}
		sos << std::endl << std::flush;
		syncout::print(os,sos);
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class FILTERCRITERION>
bool Sequential_Step_Hybrid<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,FILTERCRITERION>::evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level, std::ostream& os)
{
	assert(sub);
	assert(crit);
	assert(_filtercrit);
	assert(0<=_keep_perc && _keep_perc<=100);
	assert(_keep_max>=0);
	assert(_generalization_level>=1 && _generalization_level<=sub->get_n());
	track=parent::result_tracker_active();

	if(!filter_candidates(sub,_generalization_level,os)) {if(parent::output_normal()) {std::ostringstream sos; sos << "FAILURE filter_candidates() sub=" << *sub << std::endl << std::endl << std::flush; syncout::print(os,sos);} return false;}
	assert(_filterlist.size()>0);

	if(_keep_perc==0) { // no further testing, return the best identified during filtering
		assert(_filterlist.size()>0);
		sub->stateless_copy( *((_filterlist.begin())->_sub) );
		assert(sub->get_d()!=0);
		if(!crit->evaluate(result,sub)) {if(parent::output_normal()) {std::ostringstream sos; sos << "FAILURE crit->eval() sub=" << *(sub) << std::endl << std::endl << std::flush; syncout::print(os,sos);} return false;}
		if(track) parent::_tracker->add(result,sub);
		return true;
	}

	if(!tmp_step_sub || tmp_step_sub->get_n()<sub->get_n()) tmp_step_sub.reset(new SUBSET(sub->get_n()));
	RETURNTYPE val,bestval;
	iter=_filterlist.begin();
	if(!crit->evaluate(bestval,iter->_sub)) {if(parent::output_normal()) {std::ostringstream sos; sos << "FAILURE 1st crit->eval() sub=" << *(iter->_sub) << std::endl << std::endl << std::flush; syncout::print(os,sos);} return false;}
	if(track) parent::_tracker->add(bestval,iter->_sub);
	tmp_step_sub->stateless_copy(*(iter->_sub)); // store first candidate as the initially best
	iter++;
	while(iter!=_filterlist.end()) {
		if(!crit->evaluate(val,iter->_sub)) {if(parent::output_normal()) {std::ostringstream sos; sos << "FAILURE crit->eval() sub=" << *(iter->_sub) << std::endl << std::endl << std::flush; syncout::print(os,sos);} return false;}
		if(track) parent::_tracker->add(val,iter->_sub);
		if(val>bestval) {
			bestval=val;
			tmp_step_sub->stateless_copy(*(iter->_sub));
		}
		iter++;
	}
	sub->stateless_copy(*tmp_step_sub);
	assert(sub->get_d()!=0);
	result=bestval;

	return true;
}


} // namespace
#endif // FSTSEARCHSEQSTEPHYBRID_H ///:~
