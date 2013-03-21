#ifndef FSTSEARCHSEQSFS_H
#define FSTSEARCHSEQSFS_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    search_seq_sfs.hpp
   \brief   implements Sequential_Forward_Selection and Sequential_Backward_Selection
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
#include "error.hpp"
#include "global.hpp"
#include "stopwatch.hpp"
#include "search_seq.hpp"

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

/*! \brief Implements Sequential_Forward_Selection and Sequential_Backward_Selection

	\note By default any initial subset contents are ignored; SFS starts from empty set and SBS starts from full set. 
	      Call enable_initial_subset() to let the search start from the initial sub contents.
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
class Search_SFS : public Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>{ 
public:
	typedef Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR> parent;
	typedef typename parent::parent grandparent;
	typedef boost::shared_ptr<EVALUATOR> PEvaluator;
	typedef boost::shared_ptr<CRITERION> PCriterion;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Search_SFS(const PEvaluator evaluator) : Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>(evaluator) {_direction=FORWARD; notify("Search_SFS constructor.");}
	virtual ~Search_SFS() {notify("Search_SFS destructor.");}
	
	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout); //!< returns found subset + criterion value

	void set_search_direction(const SearchDirection direction) {assert(direction==FORWARD || direction==BACKWARD); _direction=direction;} //!< FORWARD implies SFS, BACKWARD implies SBS (do not change concurrently to running search())
	SearchDirection get_search_direction() const {return _direction;}
	
	bool get_result(const DIMTYPE d, RETURNTYPE &result, PSubset &sub); //!< retrieves temporary solution of cardinality d that had been updated in the course of the last search (if it exists)

	virtual std::ostream& print(std::ostream& os) const;
protected:
	SearchDirection _direction; //!< accepted values FORWARD for SFS or BACKWARD for SBS
	//! Structure to hold [subset,criterion value] temporary solutions in the course of search
	struct OneSubset {
		RETURNTYPE critvalue; 
		boost::scoped_ptr<SUBSET> sub;
	};
	boost::scoped_array<OneSubset> bsubs;
	DIMTYPE _n; //!< bsubs array size
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
std::ostream& Search_SFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::print(std::ostream& os) const 
{
	if(parent::get_generalization_level()>1) os << "(G)";
	if(_direction==FORWARD) os << "SFS"; else os << "SBS";
	if(parent::get_generalization_level()>1) os << ", G=" << parent::get_generalization_level();
	os << "  [Search_SFS()";
	if(parent::_evaluator) os << " with " << *parent::_evaluator; 
	if(grandparent::result_tracker_active()) os << " with " << *grandparent::_tracker; 
	os << "]";
	return os;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_SFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	// NOTE: certain parameter combinations (initial sub->get_d() versus target_d versus parent::_generalization_level) 
	//       make it impossible to yield a valid solution. In such case search() just returns false
	StopWatch swatch;
	if(parent::_evaluator) parent::_evaluator->set_output_detail(grandparent::get_output_detail()); else throw FST::fst_error("Search_Sequential: Missing evaluator.");
	bool track=grandparent::result_tracker_active(); if(track) grandparent::_tracker->set_output_detail(grandparent::get_output_detail());
	assert(sub);
	assert(crit);
	assert(sub->get_n()>0);
	assert(target_d>=0 && target_d<=sub->get_n());
	assert(parent::get_generalization_level()>=1 && parent::get_generalization_level()<=sub->get_n());

	const bool _forward=(_direction==FORWARD);
	if(!parent::initial_subset_enabled()) {
		if(_forward) sub->deselect_all(); else sub->select_all();
	}
	DIMTYPE _d=sub->get_d();
	_n=sub->get_n();
	bsubs.reset(new OneSubset[_n]);

	if( (_forward && _d+parent::get_generalization_level()>_n) || 
		(!_forward && _d<=parent::get_generalization_level()) || 
		(target_d>0 && _forward && _d>=target_d) || 
		(target_d>0 && !_forward && _d<=target_d) ||
		(target_d>0 && ((target_d>_d)?(target_d-_d):(_d-target_d))%parent::get_generalization_level()>0) ) return false; // no solution reachable under this setting
	
	if(grandparent::output_normal()) {
		std::ostringstream sos; 
		sos << "---------------------------------------" << std::endl;
		sos << "Starting " << *this << std::endl;
		sos << "with Criterion: " << *crit << std::endl;
		sos << "with initial subset: " << *sub << std::endl;
		if(target_d==0) sos << "Subset size to be optimized." << std::endl; else sos << "Target subset size set to: " << target_d << std::endl;
		sos << std::endl << std::flush;
		syncout::print(os,sos);
	}

	sub->set_forward_mode(_forward); // FORWARD is the prevailing mode

	if(_d>0) { // evaluate initial subset if any
		if(!crit->evaluate(result,sub)) return false; if(track) grandparent::_tracker->add(result,sub);
		bsubs[_d-1].sub.reset(new SUBSET(_n));
		bsubs[_d-1].sub->stateless_copy(*sub);
		bsubs[_d-1].critvalue=result;
		if(grandparent::output_normal()) {std::ostringstream sos; sos << "subresult=" << result << " " << *sub << std::endl << std::endl; syncout::print(os,sos);}
	}

	DIMTYPE steps;
	if(target_d>0 && target_d<=_n) { // search for subset of size target_d
		steps=((target_d>_d)?(target_d-_d):(_d-target_d))/parent::get_generalization_level();
	} else { // find best dimensionality
		if(_forward) steps=(_n-_d)/parent::get_generalization_level();
		else steps=(_d-1)/parent::get_generalization_level();
	}
	
	// main search loop
	for(DIMTYPE i=0;i<steps;i++) {
		if(!parent::Step(_forward, result, sub, crit, os)) return false; if(track) grandparent::_tracker->add(result,sub);
		if(grandparent::output_normal()) {std::ostringstream sos; sos << "subresult=" << result << " " << *sub << std::endl << std::endl << std::flush; syncout::print(os,sos);}
		_d=sub->get_d();
		bsubs[_d-1].sub.reset(new SUBSET(_n));
		bsubs[_d-1].sub->stateless_copy(*sub);
		bsubs[_d-1].critvalue=result;
	}
	
	DIMTYPE finalidx;
	if(target_d>0 && target_d<=_n) finalidx=target_d-1;
	else {
		for(finalidx=0;finalidx<_n && !bsubs[finalidx].sub;finalidx++);
		for(DIMTYPE i=finalidx+1;i<_n;i++) if(bsubs[i].sub && bsubs[i].critvalue>bsubs[finalidx].critvalue) finalidx=i;
	}

	if(finalidx>=_n || !bsubs[finalidx].sub) return false;

	result=bsubs[finalidx].critvalue;
	sub->stateless_copy(*(bsubs[finalidx].sub));

	if(grandparent::output_normal()) {
		std::ostringstream sos; 
		sos << "......................................." << std::endl;
		sos << "Search_SFS() search finished. " << swatch << std::endl;
		sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
		syncout::print(os,sos);
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_SFS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::get_result(const DIMTYPE d, RETURNTYPE &result, PSubset &sub) // retrieves temporary solution of cardinality d that had been updated in the course of the last search
{
	assert(sub);
	assert(d>0 && d<=_n);
	if(bsubs && d>0 && d<=_n && bsubs[d-1].sub) {
		result=bsubs[d-1].critvalue;
		sub->stateless_copy(*(bsubs[d-1].sub));
		return true;
	}
	return false;
}


} // namespace
#endif // FSTSEARCHSEQSFS_H ///:~
