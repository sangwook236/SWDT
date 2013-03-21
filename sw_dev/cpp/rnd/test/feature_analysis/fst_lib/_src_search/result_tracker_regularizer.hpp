#ifndef FSTRESULTTRACKERREGULARIZER_H
#define FSTRESULTTRACKERREGULARIZER_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    result_tracker_regularizer.hpp
   \brief   Enables eventual selection of a different subset
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
#include <list>
#include "error.hpp"
#include "global.hpp"
#include "result_tracker_dupless.hpp"

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

/*! \brief Collects multiple results. Enables eventual selection of alternative solution based on secondary criterion.

    Enables selecting a different subset than the one
    yielded by the search procedure, e.g., with the aim to reduce overfitting
    or to take into account known feature acquisition cost. 
    
    The idea is to consider all known
    subsets with criterion value close enough to the maximum as effectively
    equal - then another criterion is used to choose among those alternative
    solutions. This technique can improve generalization/robustness and reduce
    overfitting, as well as open up other possibilities, like preferring
    a subset being almost equal to the primarily selected, but with considerably 
    lower feature value acquisition cost, etc. For details see paper
    "Somol, Grim, Pudil: The Problem of Fragile Feature Subset Preference in Feature Selection Methods 
    and A Proposal of Algorithmic Workaround. In Proc. ICPR 2010.  IEEE Computer 
    Society, 2010".

*/ 
template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Result_Tracker_Regularizer : public Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> {
public:
	typedef Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> parent;
	typedef typename parent::ResultRec ResultRec;
	typedef ResultRec* PResultRec;
	typedef boost::shared_ptr<SUBSET> PSubset;
	typedef boost::shared_ptr<CRITERION> PCriterion;
	Result_Tracker_Regularizer(const IDXTYPE capacity_limit=0):Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>(capacity_limit),_searching(false) {notify("Result_Tracker_Regularizer constructor.");}
	Result_Tracker_Regularizer(const Result_Tracker_Regularizer& rtr):Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>(rtr),_searching(false) {notify("Result_Tracker_Regularizer copy-constructor.");}
	virtual ~Result_Tracker_Regularizer() {notify("Result_Tracker_Regularizer destructor.");}
	
	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool optimize_within_margin(const RETURNTYPE margin, RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub, const PCriterion crit); // returns subset best among those within margin (margin with respect to criterion optimized before when bulding tracker list) according to criterion crit

	virtual bool optimize_within_first_equivalence_group(RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub, const PCriterion crit); //!< returns the subset from the first equivalence group (=subsets with equal primary criterion value) yielding the highest crit value
	virtual bool optimize_within_next_equivalence_group(RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub); //!< returns the subset from the next equivalence group (=subsets with equal primary criterion value which is the next highest in sequence) yielding the highest crit value

	Result_Tracker_Regularizer* clone() const {throw fst_error("Result_Tracker_Regularizer::clone() not supported, use Result_Tracker_Regularizer::stateless_clone() instead.");}
	Result_Tracker_Regularizer* sharing_clone() const {throw fst_error("Result_Tracker_Regularizer::sharing_clone() not supported, use Result_Tracker_Regularizer::stateless_clone() instead.");}
	Result_Tracker_Regularizer* stateless_clone() const;

	virtual std::ostream& print(std::ostream& os) const {os << "Result_Tracker_Regularizer(limit=" << parent::_capacity_limit << ", margin=" << parent::_margin << ") size " << parent::results.size(); return os;}

private:
	bool optimize_within_equivalence_group(RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub); //!< building block for optimize_within_*_equivalence_group
	PCriterion _crit;
	RETURNTYPE _prevmax;
	typename parent::CONSTRESULTSITER _iter;
	bool _searching;
};

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
Result_Tracker_Regularizer<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET, CRITERION>* Result_Tracker_Regularizer<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET, CRITERION>::stateless_clone() const
{
	Result_Tracker_Regularizer<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET, CRITERION> *clone=new Result_Tracker_Regularizer<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET, CRITERION>(*this);
	clone->set_cloned();
	return clone;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Result_Tracker_Regularizer<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET, CRITERION>::optimize_within_margin(const RETURNTYPE margin, RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub, const PCriterion crit)
{
	assert(margin>=0);
	if(parent::results.empty()) return false;
	bool stop=false;
	
	typename parent::CONSTRESULTSITER iter=parent::results.begin();
	typename parent::CONSTRESULTSITER sel=iter;
	RETURNTYPE bestval, val;
	
	// NOTE: crit is expected to be already initialize()d before use here
	if(!crit->evaluate(bestval,sel->sub)) return false;
	notify("sel: val1=",sel->value,", val2=",bestval,", getd ",sel->sub->get_d(),".");
	
	const RETURNTYPE threshold=sel->value-margin;
	iter++;
	while(!stop && iter!=parent::results.end())
	{
		if(iter->value<threshold) stop=true; else
		{
			// NOTE: crit is expected to be already initialize()d before use here
			if(!crit->evaluate(val,iter->sub)) return false;
			notify("iter: val1=",iter->value,", val2=",val,", getd ",iter->sub->get_d(),".");
			if(val>bestval || (val==bestval && sel->sub->get_d()>iter->sub->get_d())) {bestval=val; sel=iter; notify("best updated.");}
			iter++;
		}
	}
	result1=sel->value; result2=bestval; sub=sel->sub;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Result_Tracker_Regularizer<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET, CRITERION>::optimize_within_equivalence_group(RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub)
{
	// assert _iter points to valid structure
	_prevmax=_iter->value;
	RETURNTYPE val, maxval1, maxval2; PSubset maxsub=_iter->sub;
	maxval1=_iter->value;
	if(!_crit->evaluate(maxval2,_iter->sub)) {_searching=false; return false;}
	_iter++;
	while(_iter!=parent::results.end() && _iter->value==_prevmax)
	{
		if(!_crit->evaluate(val,_iter->sub)) {_searching=false; return false;}
		if(val>maxval2 || (val==maxval2 && maxsub->get_d()>_iter->sub->get_d())) {maxval2=val; maxsub=_iter->sub;}
		_iter++;
	}
	if(_iter==parent::results.end()) _searching=false; else _searching=true;
	result1=maxval1; result2=maxval2; sub=maxsub;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Result_Tracker_Regularizer<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET, CRITERION>::optimize_within_first_equivalence_group(RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub, const PCriterion crit)
{ // returns the subset from the first equivalence group (=subsets with equal primary criterion value) yielding the highest crit value
	assert(crit);
	assert(sub);
	if(parent::results.empty()) {_searching=false; return false;}
	_crit=crit;
	_iter=parent::results.begin();
	return optimize_within_equivalence_group(result1,result2,sub);
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Result_Tracker_Regularizer<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET, CRITERION>::optimize_within_next_equivalence_group(RETURNTYPE &result1, RETURNTYPE &result2, PSubset &sub) 
{ // returns the subset from the next equivalence group (=subsets with equal primary criterion value which is the next highest in sequence) yielding the highest crit value
	assert(_crit);
	assert(sub);
	if(!_searching) return false;
	return optimize_within_equivalence_group(result1,result2,sub);
}

} // namespace
#endif // FSTRESULTTRACKERREGULARIZER_H ///:~
