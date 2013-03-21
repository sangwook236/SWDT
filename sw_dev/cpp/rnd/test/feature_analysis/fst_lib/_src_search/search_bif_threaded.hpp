#ifndef FSTSEARCHBIFTHREADED_H
#define FSTSEARCHBIFTHREADED_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    search_bif_threaded.hpp
   \brief   Threaded implementation of Best Individual Features, i.e., individual feature ranking
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
#include "stopwatch.hpp"
#include "search.hpp"
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

//! Implements Best Individual Features, i.e., individual feature ranking
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads=2>
class Search_BIF_Threaded : public Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent;
	typedef boost::shared_ptr<CRITERION> PCriterion;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Search_BIF_Threaded():Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>() {BOOST_STATIC_ASSERT(max_threads>0); notify("Search_BIF_Threaded constructor.");}
	virtual ~Search_BIF_Threaded() {notify("Search_BIF_Threaded destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout); //!< returns found subset + criterion value
	bool evaluate_individuals(const DIMTYPE n, const PCriterion crit, std::ostream& os=std::cout);

	bool getFirstBIF(RETURNTYPE &value, DIMTYPE &feature) const;
	bool getNextBIF(RETURNTYPE &value, DIMTYPE &feature) const;
	
	virtual std::ostream& print(std::ostream& os) const {os << "Individual feature ranking  [Search_BIF_Threaded(threads="<<max_threads<<")"; if(parent::result_tracker_active()) os << " with " << *parent::_tracker; os<<"]"; return os;}
protected:
	//! Structure to hold [feature,criterion value] pair while ranking features in Search_BIF_Threaded
	typedef struct {DIMTYPE feature; RETURNTYPE critval;} OneFeature;
	typedef std::list<OneFeature> FEATURELIST;
	FEATURELIST bifs;
protected:
	Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads> _evaluator;
private:
	mutable typename FEATURELIST::const_iterator get_iter; //!< get*BIF() support
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
bool Search_BIF_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::evaluate_individuals(const DIMTYPE n, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	notify("Search_BIF_Threaded::evaluate_indviduals().");
	assert(crit);
	assert(n>0);
	OneFeature ofbuf;
	boost::shared_ptr<SUBSET> tempsub(new SUBSET(n)); // tempsub forward_mode assumed
	tempsub->deselect_all();
	typename FEATURELIST::iterator iter;
	RETURNTYPE val;
	DIMTYPE i;
	bifs.clear();
	_evaluator.initialize(crit, n);
	
	if(parent::output_normal()) {std::ostringstream sos; sos << "Evaluating feature: "; syncout::print(os,sos);}
	for(i=0;i<n;i++) { // compute individual values, store in list
		if(i>0) tempsub->deselect(i-1);
		tempsub->select(i);
		if(parent::output_normal()) {std::ostringstream sos; sos <<i<<", "<<std::flush; syncout::print(os,sos);}
		_evaluator.add_candidate(tempsub,i);
	}
	if(parent::output_normal()) {std::ostringstream sos; sos << " ...finished." << std::endl<<std::endl<<std::flush; syncout::print(os,sos);}
		
	PSubset psub;
	DIMTYPE count=0;
	for(bool got=_evaluator.getFirstResult(val, psub, i);got;got=_evaluator.getNextResult(val, psub, i))
	{
		count++;
		assert(psub);
		assert(psub->get_n()==n);
		assert(psub->get_d()==1);
		
		ofbuf.feature=i; ofbuf.critval=val;
		for(iter=bifs.begin(); iter!=bifs.end() && ofbuf.critval<(*iter).critval; iter++) {}
		bifs.insert(iter,ofbuf);

		if(parent::output_normal()) {std::ostringstream sos; sos << "f:"<<i<<", val="<<val<<std::endl<<std::flush; syncout::print(os,sos);}
	}
	if(parent::output_normal()) {std::ostringstream sos; sos <<std::endl<<std::flush; syncout::print(os,sos);}
	assert(count==n);
	if(parent::output_detailed()) {
		std::ostringstream sos; sos << "bifs: ";
		for(iter=bifs.begin(); iter!=bifs.end(); iter++) sos << (*iter).feature << " "; sos << std::endl <<std::endl<<std::flush;
		syncout::print(os,sos);
	}
	return true;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
bool Search_BIF_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::getFirstBIF(RETURNTYPE &value, DIMTYPE &feature) const
{
	get_iter = bifs.begin();
	if(get_iter!=bifs.end()) {
		value=(*get_iter).critval;
		feature=(*get_iter).feature;
		get_iter++;
		return true;
	} else return false;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
bool Search_BIF_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::getNextBIF(RETURNTYPE &value, DIMTYPE &feature) const
{
	if(get_iter!=bifs.end()) {
		value=(*get_iter).critval;
		feature=(*get_iter).feature;
		get_iter++;
		return true;
	} else return false;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
bool Search_BIF_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + criterion value
{
	StopWatch swatch;
	bool track=parent::result_tracker_active(); if(track) parent::_tracker->set_output_detail(parent::get_output_detail());
	notify("Search_BIF_Threaded::search().");
	assert(sub);
	assert(crit);
	assert(sub->get_n()>0);
	assert(target_d>=0 && target_d<sub->get_n());
	DIMTYPE _n=sub->get_n();

	if(parent::output_normal()) {
		std::ostringstream sos; 
		sos << "---------------------------------------" << std::endl;
		sos << "Starting " << *this << std::endl;
		sos << "with Criterion: " << *crit << std::endl;
		if(target_d==0) sos << "Subset size to be optimized." << std::endl; else sos << "Target subset size set to: " << target_d << std::endl;
		sos << std::endl << std::flush;
		syncout::print(os,sos);
	}

	if(!evaluate_individuals(_n,crit,os)) return false;

	boost::shared_ptr<SUBSET> tempsub(new SUBSET(_n));
	typename FEATURELIST::const_iterator iter;
	DIMTYPE i;	
	if(target_d>0) { // return subset of size target_d
		sub->deselect_all();
		for(iter=bifs.begin(),i=0; iter!=bifs.end() && i<target_d; iter++,i++) sub->select((*iter).feature);
		if(!crit->evaluate(result,sub)) return false; if(track) parent::_tracker->add(result,sub);
	}
	else { // find best dimensionality
		tempsub->deselect_all();
		
		iter=bifs.begin(); // initial subset
		assert(iter!=bifs.end());

		tempsub->select((*iter).feature);
		_evaluator.add_candidate(tempsub);
		iter++;
		while(iter!=bifs.end()) { // any better option ?
			tempsub->select((*iter).feature);
			_evaluator.add_candidate(tempsub);
			iter++;
		}
		if(track) _evaluator.results_to_tracker(parent::_tracker);
		PSubset psub;
		DIMTYPE dummyfeature;
		if(!_evaluator.getBestResult(result,psub,dummyfeature)) return false;
		assert(psub);
		assert(psub->get_n()==sub->get_n());
		sub->stateless_copy(*psub);
	}
	if(parent::output_normal()) {
		std::ostringstream sos; 
		sos << "......................................." << std::endl;
		sos << "Search_BIF_Threaded() search finished. " << swatch << std::endl;
		sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
		syncout::print(os,sos);
	}
	return true;
}


} // namespace
#endif // FSTSEARCHBIFTHREADED_H ///:~
