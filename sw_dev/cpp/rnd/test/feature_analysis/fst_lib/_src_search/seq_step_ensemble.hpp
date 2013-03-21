#ifndef FSTSEARCHSEQSTEPENSEMBLE_H
#define FSTSEARCHSEQSTEPENSEMBLE_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    seq_step_ensemble.hpp
   \brief   Implements voting ensemble selection step in sequential search type of methods
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
#include <vector>
#include <map>
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

//----------------------------------------------------------------------------

/*! \brief Implements voting ensemble selection step in sequential search type of methods
to possibly improve robustness and stability of feature selection result.

\detailed Criteria ensembles may improve generalization and stability properties of the selected feature subset, see 
"P. Somol, J. Grim,  and P. Pudil.  Criteria Ensembles in Feature Selection.  In Proc. MCS, LNCS 5519, pages 304–313.  Springer, 2009."
The idea is to reduce possible over-training, i.e., excessive result adjustment to particular
criterion properties. By employing more different criteria the result is likely to be
more robust in different contexts. Criteria ensembles as implemented here
are based on voting about feature preferences. In sequential algorithm step, feature
candidates are ordered separatly according to each considered criterion. The various
orderings are then joined (feature position index averaged) to produce final feature
ordering. The best feature is then selected for addition to the current working subset.
Note that this mechanism allows to use completely unrelated criteria, as the various criterion
values are never combined - the only information that is combined is position index in ordered
feature lists. This is advantageous as it enables combinations of filter and wrapper criteria
(that yield values from different intervals) etc. Note, however, that the value
obtained as result of ensemble evaluation is not usable for assessing the whole feature
subset - it can be used only within the selection step to identify one feature
(can be extended to feature c-tuple) for next inclusion/removal. For this reason
it is also necessary to employ one single criterion to be used by selection algorithm
for evaluating current subsets (and thus directing next search steps). This single
criterion is used in the same way as in other FST3 Sequential_Step implementations,
i.e. it must be passed in evaluate_candidates call. The criteria that form the ensemble
are to be passed to Sequential_Step_Ensemble constructor.

\note Single feature ordering implementation only. Support for generalized search 
      (evaluation of candidate c-tuples instead of single features) may be implemented in future,
      but so far is not. Thus, _generalization_level is ignored.
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Sequential_Step_Ensemble : public Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent;
	typedef boost::shared_ptr<CRITERION> PCriterion;
	typedef boost::shared_ptr<Criterion<RETURNTYPE,SUBSET> > PAbstractCriterion;
	typedef std::vector<PAbstractCriterion> PAbstractCriteria;
	typedef boost::shared_ptr<PAbstractCriteria> PEnsembleCriteria;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Sequential_Step_Ensemble(const PEnsembleCriteria ensemble):Sequential_Step<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(),track(false) {assert(ensemble); _ensemble=ensemble; notify("Sequential_Step_Ensemble constructor.");}
	virtual ~Sequential_Step_Ensemble() {notify("Sequential_Step_Ensemble destructor.");}

	// NOTE: crit (as well as all criteria in ensemble) is expected to be already initialize()d before use here
	inline bool evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level=1, std::ostream& os=std::cout); //!< chooses among subsets offered by sub->get*CandidateSubset()

	virtual std::ostream& print(std::ostream& os) const {os << "Sequential_Step_Ensemble()"; if(parent::result_tracker_active()) os << " with " << *parent::_tracker; return os;}
protected:
	inline bool order_candidates(const PSubset sub);
	inline bool test_candidate(const PSubset sub);
	
	PEnsembleCriteria _ensemble;
	
	//! Nested class to hold feature/subset candidate info in the course of the ensemble voting process
	class SubsetCandidate {
	public:
		SubsetCandidate(const RETURNTYPE value, const PSubset sub, const DIMTYPE tfeature) : _value(value), _sub(new SUBSET(*sub)), _tempfeature(tfeature) {}
		SubsetCandidate(const SubsetCandidate& fc) : _value(fc._value), _sub(fc._sub), _tempfeature(fc._tempfeature) {}
		RETURNTYPE _value;
		const PSubset _sub;
		const DIMTYPE _tempfeature; // to hold copy of sub->getFirstTemporaryFeature()
	};
	typedef boost::shared_ptr<SubsetCandidate> PSubsetCandidate;
	typedef std::list<SubsetCandidate> CANDIDATELIST;
	typedef boost::shared_ptr<CANDIDATELIST> PCANDIDATELIST;
	typedef std::vector<PCANDIDATELIST> CANDIDATELISTS;
	CANDIDATELISTS _lists; 
	typedef std::map<DIMTYPE, PSubsetCandidate> FINALVALUES;
	FINALVALUES _final;
	
	typename PAbstractCriteria::iterator citer; // iterate vector of criteria
	typename CANDIDATELISTS::iterator oiter; // iterate lists of subset candidates
	typename CANDIDATELIST::iterator iter; // iterate lists of subset candidates
	typename FINALVALUES::iterator fiter; // iterate lists of subset candidates

private:
	boost::scoped_ptr<SUBSET> tmp_step_sub; // (lazy allocation)
	bool track;
};


template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Sequential_Step_Ensemble<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::test_candidate(const PSubset sub)
{
	assert(sub);
	assert(sub->get_d()!=0);
	assert(sub->getNoOfTemporaryFeatures()>0);
	assert(_ensemble);
	assert(!_ensemble->empty());

	RETURNTYPE val;
	DIMTYPE _tfeat;
	assert(_ensemble->size()==_lists.size());
	for(citer=_ensemble->begin(), oiter=_lists.begin(); citer!=_ensemble->end() && oiter!=_lists.end(); citer++, oiter++)
	{
		if(!(*citer)->evaluate(val,sub)) return false; 
		for(iter=(*oiter)->begin(); iter!=(*oiter)->end() && val<iter->_value; iter++); 
		if(!sub->getFirstTemporaryFeature(_tfeat)) return false; // just one exists due to getFirstCandidateSubset(1,false/*reverse*/) call in order_candidates()
		(*oiter)->insert(iter,SubsetCandidate(val,sub,_tfeat));
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Sequential_Step_Ensemble<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::order_candidates(const PSubset sub)
{
	assert(sub);
	
	_final.clear();
	
	// first build ordered candidate lists (_lists) 
	bool b=sub->getFirstCandidateSubset(1,false/*reverse*/); if(!b) return false; // no candidate subsets can be traversed for whatever reason
	
	DIMTYPE _tfeat;
	if(!sub->getFirstTemporaryFeature(_tfeat)) return false; // just one exists due to getFirstCandidateSubset(1,false/*reverse*/) call in order_candidates()
	assert(_tfeat>=0 && _tfeat<sub->get_n());
	PSubsetCandidate _cs(new SubsetCandidate(0.0,sub,_tfeat));
	_final.insert( make_pair(_tfeat,_cs));
		
	if(!test_candidate(sub)) return false;
	b=sub->getNextCandidateSubset(); // evaluate the remaining candidates
	while(b) {
		if(!sub->getFirstTemporaryFeature(_tfeat)) return false; // just one exists due to getFirstCandidateSubset(1,false/*reverse*/) call in order_candidates()
		assert(_tfeat>=0 && _tfeat<sub->get_n());
		PSubsetCandidate _cs(new SubsetCandidate(0.0,sub,_tfeat));
		_final.insert( make_pair(_tfeat,_cs));

		if(!test_candidate(sub)) return false;
		b=sub->getNextCandidateSubset();
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Sequential_Step_Ensemble<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::evaluate_candidates(RETURNTYPE &result, const PSubset sub, const PCriterion crit, const DIMTYPE _generalization_level, std::ostream& os)
{
	assert(sub);
	assert(sub->get_n()>0);
	assert(crit);
	assert(_ensemble);
	assert(!_ensemble->empty());
	if(_generalization_level!=1) throw FST::fst_error("Sequential_Step_Ensemble currently does not support generalization level other than 1.");
	track=parent::result_tracker_active();
	
	_lists.reserve(_ensemble->size());
	while(_lists.size()<_ensemble->size()) _lists.push_back(PCANDIDATELIST(new CANDIDATELIST));
	assert(_lists.size()==_ensemble->size());
	for(oiter=_lists.begin(); oiter!=_lists.end(); oiter++) (*oiter)->clear();

	if(!order_candidates(sub)) {if(parent::output_normal()) {std::ostringstream sos; sos << "FAILURE order_candidates() sub=" << *sub << std::endl << std::endl << std::flush; syncout::print(os,sos);} return false;}

	assert(_final.size()>0);
	for(fiter=_final.begin();fiter!=_final.end();fiter++) (*fiter).second->_value=0.0;

	// collect ordering votes
	RETURNTYPE _lastval;
	DIMTYPE _curposition;
	for(oiter=_lists.begin(); oiter!=_lists.end(); oiter++)
	{
		iter=(*oiter)->begin();
		_curposition=1;
		_lastval=iter->_value;
		_final[iter->_tempfeature]->_value+=_curposition;
		iter++;
		while(iter!=(*oiter)->end())
		{
			if(iter->_value<_lastval) {_curposition++; _lastval=iter->_value;}
			_final[iter->_tempfeature]->_value+=_curposition;
			iter++;
		}
	}
	
	if(!tmp_step_sub || tmp_step_sub->get_n()<sub->get_n()) tmp_step_sub.reset(new SUBSET(sub->get_n()));
	
	_lastval=0.0;
	_curposition=0;
	for(fiter=_final.begin();fiter!=_final.end();fiter++) 
	{
		if(_lastval<1.0 || ( (*fiter).second->_value < _lastval && (*fiter).second->_value > 0.0 ))
		{
			_lastval=(*fiter).second->_value; 
			_curposition=(*fiter).second->_tempfeature;
			tmp_step_sub->stateless_copy(*((*fiter).second->_sub)); // store first candidate as the initially best
		}
	}
	sub->stateless_copy(*tmp_step_sub);
	assert(sub->get_d()!=0);
	
	if(!crit->evaluate(result,sub)) {if(parent::output_normal()) {std::ostringstream sos; sos << "FAILURE crit->eval() sub=" << *(sub) << std::endl << std::endl << std::flush; syncout::print(os,sos);} return false;}
	if(track) parent::_tracker->add(result,sub);

	if(parent::output_detailed()) {
		std::ostringstream sos; 
		sos << "---------------- "<< std::endl;
		sos << "candidate lists: "<< std::endl;
		int lst=0;
		for(oiter=_lists.begin(); oiter!=_lists.end(); oiter++)
		{
			sos << lst++ << ": ";
			for(iter=(*oiter)->begin();iter!=(*oiter)->end();iter++) sos << iter->_tempfeature << "," <<iter->_value << " | ";
			sos << std::endl;
		}
		sos << "final list: "<<std::endl;
		sos << "   ";
		assert(_final.size()>0);
	
		for(fiter=_final.begin();fiter!=_final.end();fiter++) sos << (*fiter).first << ") " << (*fiter).second->_value << " | ";
		sos << std::endl;
		sos << "verified crit value=" << result << std::endl;
		sos << std::endl << std::flush;	
		syncout::print(os,sos);
	}
	
	return true;
}


} // namespace
#endif // FSTSEARCHSEQSTEPENSEMBLE_H ///:~
