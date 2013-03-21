#ifndef FSTRESULTTRACKERDUPLESS_H
#define FSTRESULTTRACKERDUPLESS_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    result_tracker_dupless.hpp
   \brief   Enables collecting multiple results
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

/*! \brief Collects multiple results, avoiding duplicates. 

    See Result_Tracker description why this is needed.
    Upon adding, sorts results descending according to value, in case of
    ties smaller subset is considered better.
*/ 
template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
class Result_Tracker_Dupless : public Result_Tracker<RETURNTYPE,SUBSET> {
public:
	typedef Result_Tracker<RETURNTYPE,SUBSET> parent;
	typedef typename parent::ResultRec ResultRec;
	typedef typename parent::PResultRec PResultRec;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Result_Tracker_Dupless(const IDXTYPE capacity_limit=0):_margin(0),_restrict_to_d(0),_capacity_limit(capacity_limit) {notify("Result_Tracker_Dupless constructor.");}
	Result_Tracker_Dupless(const Result_Tracker_Dupless& rtd):Result_Tracker<RETURNTYPE,SUBSET>(rtd), _margin(rtd._margin), _restrict_to_d(rtd._restrict_to_d), _capacity_limit(rtd._capacity_limit) {notify("Result_Tracker_Dupless copy-constructor.");}
	virtual ~Result_Tracker_Dupless() {notify("Result_Tracker_Dupless destructor.");}

	virtual bool add(const RETURNTYPE value, const PSubset sub);
	virtual void clear() {results.clear();}
	virtual long size() const {return results.size();}
	virtual long size(const RETURNTYPE margin) const;
	virtual bool get_first(PResultRec &r); //!< iterator
	virtual bool get_next(PResultRec &r); //!< iterator
	virtual void join(Result_Tracker<RETURNTYPE,SUBSET> &src); //!< adds contents from another tracker to this one

	        void set_inclusion_margin(const RETURNTYPE margin=0) {_margin=margin;} //!< restrict adding to only subsets with criterion value within given distance (margin) from best known
	        void set_inclusion_cardinality_restriction(const DIMTYPE d) {_restrict_to_d=d;} //!< for d>0 restricts adding only to subsets of size d, for d=0 subset size restriction is removed
	
	Result_Tracker_Dupless* clone() const {throw fst_error("Result_Tracker_Dupless::clone() not supported, use Result_Tracker_Dupless::stateless_clone() instead.");}
	Result_Tracker_Dupless* sharing_clone() const {throw fst_error("Result_Tracker_Dupless::sharing_clone() not supported, use Result_Tracker_Dupless::stateless_clone() instead.");}
	Result_Tracker_Dupless* stateless_clone() const;

	virtual std::ostream& print(std::ostream& os) const {os << "Result_Tracker_Dupless(limit=" << _capacity_limit << ", margin=" << _margin << ") size " << size(); return os;}
protected:
	std::list<ResultRec> results;
	RETURNTYPE _margin;
	DIMTYPE _restrict_to_d;
	const IDXTYPE _capacity_limit;

public:
	typedef typename std::list<ResultRec>::const_iterator CONSTRESULTSITER;
	typedef typename std::list<ResultRec>::iterator RESULTSITER;
	RESULTSITER begin() {return results.begin();}
	RESULTSITER end() {return results.end();}
	
private:
	RESULTSITER getiter;
};

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>* Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::stateless_clone() const
{
	Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> *clone=new Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>(*this);
	clone->set_cloned();
	return clone;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::add(const RETURNTYPE value, const PSubset sub)
{
	assert(_margin>=0);
	assert(_capacity_limit>=0);
	assert(sub);
	if(_restrict_to_d>0 && sub->get_d()!=_restrict_to_d) return false;
	ResultRec rr; rr.value=value; rr.sub.reset(new SUBSET(*sub)); //rr.sub->stateless_copy(*sub);
	bool added=false;
	if(!results.empty()) 
	{
		RESULTSITER iter=results.begin();
		if(_margin>0 && iter->value-_margin>value)
		{ // the new result is outside the _margin given by the so-far best result value. Will not be added
		} 
		else // sort-in the new result, avoiding duplicates + possibly trim the tail of list that falls below _margin, if the new result becomes the new best
		{
			if(value>iter->value) {
				results.push_front(rr); added=true;
				while(_margin>0 && results.back().value<value-_margin) results.pop_back(); // trim the tail of results that fell below _margin
				while(_capacity_limit>0 && results.size()>_capacity_limit) results.pop_back(); // if there is a limit, trim the tail so that the list size remains restricted
			} else { // sort the new result in, trimming not necessary
				bool dupl=false;
				while(iter!=results.end() && (value<iter->value || (value==iter->value && sub->get_d()>iter->sub->get_d()))) iter++;
				while(!dupl && iter!=results.end() && (value==iter->value && sub->get_d()==iter->sub->get_d())) if(sub->equivalent(*(iter->sub))) dupl=true; else iter++;
				if(!dupl) {
					results.insert(iter,rr); added=true;
					while(_capacity_limit>0 && results.size()>_capacity_limit) results.pop_back(); // if there is a limit, trim the tail so that the list size remains restricted
				}
			}
		}
	} else results.push_back(rr);
	return added;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
long Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::size(const RETURNTYPE margin) const
{
	assert(margin>=0);
	if(results.empty()) return 0;
	CONSTRESULTSITER iter=results.begin();
	long count=0;
	RETURNTYPE threshold=iter->value-margin;
	while(iter!=results.end() && iter->value>=threshold) {count++; iter++;}
	return count;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::get_first(PResultRec &r)
{
	if(results.empty()) return false;
	getiter=results.begin();
	r=&(*getiter);
	++getiter;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::get_next(PResultRec &r)
{
	if(getiter==results.end()) return false;
	r=&(*getiter);
	++getiter;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
void Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>::join(Result_Tracker<RETURNTYPE,SUBSET> &src)
{
	if(results.empty()) // simple copy
	{
		PResultRec pr;
		for(bool got=src.get_first(pr);got==true;got=src.get_next(pr)) results.push_back(*pr);
	}
	else // join
	{
		RESULTSITER iter=begin();
		PResultRec pr;
		bool got=src.get_first(pr);
		while(got)
		{
			assert(pr->sub);
			while(iter!=end() && (iter->value>pr->value || (iter->value==pr->value && iter->sub->get_d()<pr->sub->get_d()))) iter++;
			bool dupl=false;
			while(!dupl && iter!=end() && (iter->value==pr->value && iter->sub->get_d()==pr->sub->get_d())) if(pr->sub->equivalent(*(iter->sub))) dupl=true; else iter++;
			if(!dupl) results.insert(iter,*pr);
			got=src.get_next(pr);
		}
	}
	if(!results.empty()) { // make joined list coherent with _capacity_limit and _margin restrictions
		RETURNTYPE threshold=results.front().value-_margin;
		while(_capacity_limit>0 && results.size()>_capacity_limit) results.pop_back();
		while(_margin>0 && results.back().value<threshold) results.pop_back();
	}
}

} // namespace
#endif // FSTRESULTTRACKERDUPLESS_H ///:~
