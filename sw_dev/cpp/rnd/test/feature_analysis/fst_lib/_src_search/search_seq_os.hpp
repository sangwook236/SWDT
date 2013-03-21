#ifndef FSTSEARCHSEQOS_H
#define FSTSEARCHSEQOS_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    search_seq_os.hpp
   \brief   Implements Oscillating_Search
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
#include "search_bif.hpp"
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

/*! \brief Implements Oscillating_Search

	\note By default sub is expected to be initialized prior to search() call, i.e., it should contain a valid initial subset of requested size,
	      from which the search will start. If this is not the case and sub does not contain target_d features, sub contents are adjusted 
	      (sub is enlarged by adding features or shrinked by removing features based on individual 
	      feature ranking) before the search actually starts. Setting target_d to 0 leads to repeated search() calls for each possible
	      subset size, to eventually select among them the solution with highest criterion value.
	      To ensure that any initial contents of sub are completely ignored (and initial subset is composed from scratch based purely
	      on individual feature ranking), call disable_initial_subset().

   \warning Search_OS can not be used with Sequential_Step_Hybrid() parametrized _keep_perc==0 (due to initial filtercrit and crit values mix-up)
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
class Search_OS : public Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>{ 
public:
	typedef Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR> parent;
	typedef typename parent::parent grandparent;
	typedef Search_BIF<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> SearchBIF;
	typedef boost::shared_ptr<SearchBIF> PSearchBIF;
	typedef boost::shared_ptr<EVALUATOR> PEvaluator;
	typedef boost::shared_ptr<CRITERION> PCriterion;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Search_OS(const PEvaluator evaluator) : Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>(evaluator), _delta(1), track(false) {parent::enable_initial_subset(); notify("Search_OS constructor.");}
	virtual ~Search_OS() {notify("Search_OS destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout); //!< returns found subset + criterion value

	void set_delta(const DIMTYPE delta) {assert(delta>0); _delta=delta;} //!< oscillating cycle depth
	DIMTYPE get_delta() const {return _delta;}
	
	virtual std::ostream& print(std::ostream& os) const;
protected:
	enum SwingPhase {
		RESTART=0,
		DOWN=1,
		RETURN_UP=2,
		UP=4,
		RETURN_DOWN=8
	};
	
	//! standard OS initialized by sub, target dimensionality sub->get_d()
	virtual bool search_one(RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os); //!< returns found subset + criterion value
	//! if sub->get_d!=target_d, the contents of sub are either supplemented by best individual features or reduced by removing worst individual features to reach the target_d cardinality
	virtual bool adjust_initial_subset(const DIMTYPE target_d, const PSubset sub, const PSearchBIF bif, std::ostream& os);

	DIMTYPE _delta;

	boost::scoped_ptr<SUBSET> pivotsub;
	boost::scoped_ptr<SUBSET> pivot_top; //!< helper to prevent redundant search when OS cycle depth increases
	boost::scoped_ptr<SUBSET> pivot_bottom; //!< helper to prevent redundant search when OS cycle depth increases
	
	boost::scoped_ptr<SUBSET> inisub; //!< used when testing all subset sizes
	boost::shared_ptr<SUBSET> tmpsub; //!< used when testing all subset sizes
private:
	bool track;
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
std::ostream& Search_OS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::print(std::ostream& os) const 
{
	if(parent::get_generalization_level()>1) os << "(G)";
	os << "OS";
	if(parent::get_generalization_level()>1) os << ", G=" << parent::get_generalization_level();
	os << ", delta=" << _delta;
	os << "  [Search_OS()";
	if(parent::_evaluator) os << " with " << *parent::_evaluator; 
	if(grandparent::result_tracker_active()) os << " with " << *grandparent::_tracker; 
	os << "]";
	return os;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_OS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::adjust_initial_subset(const DIMTYPE target_d, const PSubset sub, const PSearchBIF bif, std::ostream& os)
{
	assert(sub);
	assert(bif);
	assert(target_d>0 && target_d<=sub->get_n());
	RETURNTYPE value;
	DIMTYPE f;
	
	if(target_d!=0 && sub->get_d()!=target_d) {
		if(grandparent::output_normal()) {std::ostringstream sos; sos << "original initial "<< *sub << std::endl << std::flush; syncout::print(os,sos);}
		if(!bif->getFirstBIF(value, f)) return false;
		assert(f>=0 && f<=sub->get_n());
		if(sub->get_d()<target_d) sub->select(f); else sub->deselect(f);
		while(sub->get_d()!=target_d)
		{
			if(!bif->getNextBIF(value, f)) return false;
			assert(f>=0 && f<=sub->get_n());
			if(sub->get_d()<target_d) sub->select(f); else sub->deselect(f);
		}
		if(grandparent::output_normal()) {std::ostringstream sos; sos << "adjusted initial "<< *sub << std::endl << std::endl << std::flush; syncout::print(os,sos);}
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_OS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	// NOTE: certain parameter combinations (initial sub->get_d() versus _delta versus target_d versus parent::_generalization_level) 
	//       make it impossible to yield a valid solution. In such case search() just returns false
	StopWatch swatch;
	if(parent::_evaluator) parent::_evaluator->set_output_detail(grandparent::get_output_detail()); else throw FST::fst_error("Search_Sequential: Missing evaluator.");
	track=grandparent::result_tracker_active(); if(track) grandparent::_tracker->set_output_detail(grandparent::get_output_detail());
	assert(sub);
	assert(crit);
	assert(sub->get_n()>0);
	assert(target_d>=0 && target_d<sub->get_n());
	assert(parent::get_generalization_level()>=1 && parent::get_generalization_level()<=sub->get_n());

	if(!parent::initial_subset_enabled()) sub->deselect_all();

	if( (_delta<parent::get_generalization_level()) ) return false; // impossible to run the search

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

	PSearchBIF bif;
	DIMTYPE _n=sub->get_n();

	if(target_d==0) { // evaluate all subset sizes
		if(!inisub || inisub->get_n()!=_n) inisub.reset(new SUBSET(_n));
		if(!tmpsub || tmpsub->get_n()!=_n) tmpsub.reset(new SUBSET(_n));
		RETURNTYPE value;
		if(!bif) {
			bif.reset(new SearchBIF());
			if(!bif->evaluate_individuals(_n, crit, os)) return false;
		}
		inisub->stateless_copy(*sub);
		for(DIMTYPE d=1;d<=_n;d++) {
			tmpsub->stateless_copy(*inisub);
			if(!adjust_initial_subset(d,tmpsub,bif,os)) return false;
			if(d<_n) {if(!search_one(value,tmpsub,crit,os)) return false;} else {if(!crit->evaluate(value,tmpsub)) return false;}
			if(d==1 || value>result) {
				result=value;
				sub->stateless_copy(*tmpsub);
				if(grandparent::output_normal()) {std::ostringstream sos; sos << "NEW MAXVAL (d="<<d<<") ="<<result<<", " << *sub << std::endl << swatch << std::endl << std::endl << std::flush; syncout::print(os,sos);}
			} else
				if(grandparent::output_normal()) {std::ostringstream sos; sos << "result (d="<<d<<") ="<<value<<", " << *tmpsub << std::endl << std::endl << std::flush; syncout::print(os,sos);}
		}
	} else { // evaluate subset of target_d features
		if(sub->get_d()!=target_d) {
			if(!bif) {
				bif.reset(new SearchBIF);
				if(!bif->evaluate_individuals(_n,crit,os)) return false;
			}
			if(!adjust_initial_subset(target_d,sub,bif,os)) return false;
		}
		if(!search_one(result,sub,crit,os)) return false;
	}
	if(grandparent::output_normal()) {
		std::ostringstream sos; 
		sos << "......................................." << std::endl;
		sos << "Search_OS() search finished. " << swatch << std::endl;
		sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
		syncout::print(os,sos);
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_OS<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::search_one(RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	notify("Search_OS::search_one().");
	assert(sub);
	assert(crit);
	assert(sub->get_n()>0);
	assert(sub->get_d()>0);
	assert(parent::get_generalization_level()>=1 && parent::get_generalization_level()<=sub->get_n());

	if( (sub->get_d()==0) ||
		(_delta<parent::get_generalization_level()) ) return false; // impossible to run the search

	DIMTYPE _n=sub->get_n();
	if(!pivotsub || pivotsub->get_n()!=_n) pivotsub.reset(new SUBSET(_n));
	if(!pivot_top || pivot_top->get_n()!=_n) pivot_top.reset(new SUBSET(_n));
	if(!pivot_bottom || pivot_bottom->get_n()!=_n) pivot_bottom.reset(new SUBSET(_n));

	RETURNTYPE bestval;
	
	// sub is assumed to contain the initial subset
	sub->set_forward_mode(true);

	// NOTE: this is not sufficient with Sequential_Step_Hybrid() parametrized _keep_perc==0	
	crit->evaluate(result,sub); // store first candidate as the initially best pivot
	if(track) grandparent::_tracker->add(result,sub);

	SwingPhase swing=RESTART;

	DIMTYPE _d=sub->get_d();
	DIMTYPE _d_piv=_d;
	DIMTYPE _cur_delta=parent::get_generalization_level();
	bool pivot_top_avail=false;
	bool pivot_bottom_avail=false;

	bool swinged=false; // to avoid redundant subset copying

	while(_cur_delta<=_delta)
		switch(swing) {
		case RESTART:
			assert(_d_piv==sub->get_d());
			pivotsub->stateless_copy(*sub); bestval=result;
			pivot_top_avail=pivot_bottom_avail=false;
			_cur_delta=parent::get_generalization_level();
			swing=DOWN;
			if(grandparent::output_normal()) {std::ostringstream sos; sos << "new subresult ="<<result<<", " << *sub << std::endl << std::endl << std::flush; syncout::print(os,sos);}
			break;
		case DOWN:
			assert(_d==sub->get_d());
			assert(sub->get_d()!=0);
			if(pivot_bottom_avail) {assert(pivot_bottom->get_d()!=0); sub->stateless_copy(*pivot_bottom); _d=sub->get_d();}
			swinged=false;
			while(_d>parent::get_generalization_level() && _d_piv<_cur_delta+_d){
				if(!parent::Step(false/*forward*/, result, sub, crit, os)) {if(grandparent::output_detailed()) {std::ostringstream sos; sos << "DOWN failure" << std::endl << std::flush; syncout::print(os,sos);} return false;}
				if(track) grandparent::_tracker->add(result,sub);
				_d=sub->get_d(); swinged=true;
			}
			if(swinged) {
				assert(_d==sub->get_d());
				if(_d>parent::get_generalization_level()) {
					pivot_bottom_avail=true;
					pivot_bottom->stateless_copy(*sub);
				}
				swing=RETURN_UP;
			} else {
				pivot_bottom_avail=false;
				swing=UP;
			}
			assert(_d==sub->get_d());
			break;
		case RETURN_UP:
			assert(_d==sub->get_d());
			while(_d<_d_piv){
				assert(_d==sub->get_d());
				if(!parent::Step(true/*forward*/, result, sub, crit, os)) {if(grandparent::output_detailed()) {std::ostringstream sos; sos << "RETURN UP failure" << std::endl << std::flush; syncout::print(os,sos);} return false;}
				if(track) grandparent::_tracker->add(result,sub);
				_d=sub->get_d();
			}
			if(result>bestval) swing=RESTART; // down-swing successful - reset
			else {
				sub->stateless_copy(*pivotsub);
				swing=UP; swinged=false;
			}
			assert(_d==sub->get_d());
			break;
		case UP:
			assert(_d==sub->get_d());
			if(pivot_top_avail) {assert(pivot_top->get_d()!=0); sub->stateless_copy(*pivot_top); _d=sub->get_d();}
			while(_d+parent::get_generalization_level()<=_n && _d<_cur_delta+_d_piv){
				assert(_d==sub->get_d());
				if(!parent::Step(true/*forward*/, result, sub, crit, os)) {if(grandparent::output_detailed()) {std::ostringstream sos; sos << "UP failure" << std::endl << std::flush; syncout::print(os,sos);} return false;}
				if(track) grandparent::_tracker->add(result,sub);
				_d=sub->get_d(); swinged=true;
			}
			if(swinged) {
				assert(_d==sub->get_d());
				if(_d+parent::get_generalization_level()<=_n) {
					pivot_top_avail=true;
					pivot_top->stateless_copy(*sub);
				}
				swing=RETURN_DOWN;
			} else {
				pivot_top_avail=false;
				swing=DOWN;
				_cur_delta+=parent::get_generalization_level();
			}
			assert(_d==sub->get_d());
			break;
		case RETURN_DOWN:
			assert(_d==sub->get_d());
			while(_d>_d_piv){
				assert(_d==sub->get_d());
				if(!parent::Step(false/*forward*/, result, sub, crit, os)) {if(grandparent::output_detailed()) {std::ostringstream sos; sos << "RETURN DOWN failure" << std::endl << std::flush; syncout::print(os,sos);} return false;}
				if(track) grandparent::_tracker->add(result,sub);
				_d=sub->get_d();
			}
			if(result>bestval) swing=RESTART; // up-swing successful - reset
			else {
				sub->stateless_copy(*pivotsub);
				swing=DOWN; swinged=false;
				_cur_delta+=parent::get_generalization_level();
			}
			assert(_d==sub->get_d());
			break;
		}

	if(pivotsub->get_d()!=0) {
		sub->stateless_copy(*pivotsub);
		result=bestval;
		return true;
	}
	return false;
}


} // namespace
#endif // FSTSEARCHSEQOS_H ///:~
