#ifndef FSTRESULTTRACKERSTABILEVAL_H
#define FSTRESULTTRACKERSTABILEVAL_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    result_tracker_stabileval.hpp
   \brief   Enables collecting multiple results, then evaluating various stability measures
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

/*! \brief Collects multiple results to evaluate various stability measures on the colleciton.
    \note Duplicates are permitted.
    \note Setting nonzero capacity limit here should be avoided unless absolutely necessary, otherwise the stability measure value computation would be based on incomplete (and possibly misleading) information
*/ 
template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
class Result_Tracker_Stability_Evaluator : public Result_Tracker<RETURNTYPE,SUBSET> {
public:
	typedef Result_Tracker<RETURNTYPE,SUBSET> parent;
	typedef typename parent::ResultRec ResultRec;
	typedef ResultRec* PResultRec;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Result_Tracker_Stability_Evaluator(const IDXTYPE capacity_limit=0):_capacity_limit(capacity_limit) {notify("Result_Tracker_Stability_Evaluator constructor.");}
	Result_Tracker_Stability_Evaluator(const Result_Tracker_Stability_Evaluator& rtse):Result_Tracker<RETURNTYPE,SUBSET>(rtse), _capacity_limit(rtse._capacity_limit) {notify("Result_Tracker_Stability_Evaluator copy-constructor.");}
	virtual ~Result_Tracker_Stability_Evaluator() {notify("Result_Tracker_Stability_Evaluator destructor.");}

	virtual bool add(const RETURNTYPE value, const PSubset sub);
	virtual void clear() {results.clear();}
	virtual long size() const {return results.size();}
	virtual bool get_first(PResultRec &r); //!< iterator
	virtual bool get_next(PResultRec &r); //!< iterator

	RETURNTYPE value_mean() const; //!< mean value of the ResultRec::value values
	RETURNTYPE value_stddev() const; //!< standard deviation of the ResultRec::value values
	RETURNTYPE size_mean() const; //!< mean value of the size of ResultRec::sub subsets
	RETURNTYPE size_stddev() const; //!< standard deviation of the size of ResultRec::sub subsets

	RETURNTYPE stability_C() const; //!< stability measure "C", cf. Somol, Novovicova: Evaluating Stability and Comparing Output..., IEEE TPAMI, November 2010
	RETURNTYPE stability_CW() const; //!< stability measure "CW", cf. Somol, Novovicova: Evaluating Stability and Comparing Output..., IEEE TPAMI, November 2010
	RETURNTYPE stability_CWrel(const DIMTYPE Ysiz) const; //!< stability measure "CW_{rel}", cf. Somol, Novovicova: Evaluating Stability and Comparing Output..., IEEE TPAMI, November 2010
	RETURNTYPE stability_SH() const; //!< stability measure "SH", cf. Somol, Novovicova, Pudil: A New Measure of Feature Selection Algorithms’ Stability, 2009 IEEE International Conference on Data Mining Workshops
	RETURNTYPE stability_PH(const DIMTYPE Ysiz) const; //!< stability measure "PH", cf. Somol, Novovicova: Evaluating Stability and Comparing Output..., IEEE TPAMI, November 2010
	RETURNTYPE stability_ATI() const; //!< stability measure "ATI", cf. A. Kalousis, J. Prados, and M. Hilario, “Stability of feature selection algorithms,” in Proc of 5th IEEE International Conference on Data Mining (ICDM 05), Houston, Texas, 2005, pp. 218–225
	RETURNTYPE stability_ANHI(const DIMTYPE Ysiz) const; //!< stability measure "ANHI", cf. Somol, Novovicova: Evaluating Stability and Comparing Output..., IEEE TPAMI, November 2010

	RETURNTYPE similarity_IC(const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const; //!< similarity measure "IC", cf. Somol, Novovicova: Evaluating Stability and Comparing Output..., IEEE TPAMI, November 2010
	RETURNTYPE similarity_ICW(const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const; //!< similarity measure "ICW", cf. Somol, Novovicova: Evaluating Stability and Comparing Output..., IEEE TPAMI, November 2010
	RETURNTYPE similarity_IATI(const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const; //!< similarity measure "IATI", cf. Somol, Novovicova: Evaluating Stability and Comparing Output..., IEEE TPAMI, November 2010
	RETURNTYPE similarity_IANHI(const DIMTYPE Ysiz, const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const; //!< similarity measure "IANHI", cf. Somol, Novovicova: Evaluating Stability and Comparing Output..., IEEE TPAMI, November 2010
	
	Result_Tracker_Stability_Evaluator* clone() const {throw fst_error("Result_Tracker_Stability_Evaluator::clone() not supported, use Result_Tracker_Stability_Evaluator::stateless_clone() instead.");}
	Result_Tracker_Stability_Evaluator* sharing_clone() const {throw fst_error("Result_Tracker_Stability_Evaluator::sharing_clone() not supported, use Result_Tracker_Stability_Evaluator::stateless_clone() instead.");}
	Result_Tracker_Stability_Evaluator* stateless_clone() const;

	virtual std::ostream& print(std::ostream& os) const {os << "Result_Tracker_Stability_Evaluator(limit=" << _capacity_limit << ") size " << size(); return os;}
protected:
	std::list<ResultRec> results;
	const IDXTYPE _capacity_limit;
	
	RETURNTYPE get_TI(const PSubset &sub1, const PSubset &sub2) const;
	RETURNTYPE get_NHI(const DIMTYPE Ysiz, const PSubset &sub1, const PSubset &sub2) const;
	DIMTYPE max_features() const; //<! returns largest subset size (sub->get_n()) in list

public:
	typedef typename std::list<ResultRec>::const_iterator CONSTRESULTSITER;
	typedef typename std::list<ResultRec>::iterator RESULTSITER;
	RESULTSITER begin() {return results.begin();}
	RESULTSITER end() {return results.end();}

private:
	RESULTSITER getiter;
};

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>* Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stateless_clone() const
{
	Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> *clone=new Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>(*this);
	clone->set_cloned();
	return clone;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::add(const RETURNTYPE value, const PSubset sub)
{
	assert(sub);
	assert(_capacity_limit>=0);
	ResultRec rr; rr.value=value; rr.sub.reset(new SUBSET(*sub)); //rr.sub->stateless_copy(*sub);
	if(!results.empty()) 
	{
		RESULTSITER iter=results.begin();
		while(iter!=results.end() && (value<iter->value || (value==iter->value && sub->get_d()>iter->sub->get_d()))) iter++;
		results.insert(iter,rr);
		while(_capacity_limit>0 && results.size()>_capacity_limit) results.pop_back(); // if there is a limit, trim the tail so that the list size remains restricted
	} else results.push_back(rr);
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::get_first(PResultRec &r)
{
	if(results.empty()) return false;
	getiter=results.begin();
	r=&(*getiter);
	++getiter;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::get_next(PResultRec &r)
{
	if(getiter==results.end()) return false;
	r=&(*getiter);
	++getiter;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::value_mean() const
{
	if(results.empty()) throw fst_error("Result_Tracker_Stability_Evaluator::value_mean() called on empty list.");
	RETURNTYPE mean=0.0;
	for(CONSTRESULTSITER iter=results.begin();iter!=results.end();iter++) mean+=iter->value;
	return mean/(RETURNTYPE)results.size();
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::value_stddev() const
{
	if(results.empty()) throw fst_error("Result_Tracker_Stability_Evaluator::value_stddev() called on empty list.");
	const RETURNTYPE mean=value_mean();
	RETURNTYPE sd=0.0;
	for(CONSTRESULTSITER iter=results.begin();iter!=results.end();iter++) sd+=(iter->value-mean)*(iter->value-mean);
	return sqrt(sd/(RETURNTYPE)results.size());
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::size_mean() const
{
	if(results.empty()) throw fst_error("Result_Tracker_Stability_Evaluator::size_mean() called on empty list.");
	RETURNTYPE mean=0.0;
	for(CONSTRESULTSITER iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); mean+=iter->sub->get_d_raw();}
	return mean/(RETURNTYPE)results.size();
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::size_stddev() const
{
	if(results.empty()) throw fst_error("Result_Tracker_Stability_Evaluator::size_stddev() called on empty list.");
	const RETURNTYPE mean=size_mean();
	RETURNTYPE sd=0.0;
	for(CONSTRESULTSITER iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); sd+=(iter->sub->get_d_raw()-mean)*(iter->sub->get_d_raw()-mean);}
	return sqrt(sd/(RETURNTYPE)results.size());
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
DIMTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::max_features() const
{
	DIMTYPE maxn=0;
	if(!results.empty())
	{
		CONSTRESULTSITER iter=results.begin();
		while(iter!=results.end()) {
			DIMTYPE ss=iter->sub->get_n_raw();
			if(ss>maxn) maxn=ss;
			iter++;
		}
	}
	return maxn;	
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stability_C() const
{
	const long size=results.size();
	if(size>1)
	{
		RETURNTYPE result=0.0;
		DIMTYPE count=0;
		for(DIMTYPE f=0;f<max_features();f++) {
			long fcnt=0;
			CONSTRESULTSITER iter;
			for(iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); if(iter->sub->selected_raw(f)) fcnt++;}
			if(fcnt>0) {
				result+=(RETURNTYPE)(fcnt-1)/(RETURNTYPE)(size-1); 
				count++;
			}
		}
		if(count==0) return 0.0;
		return result/(RETURNTYPE)count;
	} 
	else if(size==1) return 1.0;
	else return 0.0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stability_CW() const
{
	const long size=results.size();
	if(size>1)
	{
		RETURNTYPE result=0.0;
		DIMTYPE count=0;
		long fcntsum=0;
		CONSTRESULTSITER iter;
		for(iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); fcntsum+=iter->sub->get_d_raw();}
		
		for(DIMTYPE f=0;f<max_features();f++) {
			long fcnt=0;
			for(iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); if(iter->sub->selected_raw(f)) fcnt++;}
			if(fcnt>0) {
				result+=((RETURNTYPE)(fcnt)/(RETURNTYPE)(fcntsum)) * ((RETURNTYPE)(fcnt-1)/(RETURNTYPE)(size-1)); 
				count++;
			}
		}
		if(count==0) return 0.0;
		return result;
	} 
	else if(size==1) return 1.0;
	else return 0.0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stability_CWrel(const DIMTYPE Ysiz) const
{
	assert(Ysiz>0);
	const long size=results.size();
	if(size>1)
	{
		RETURNTYPE result=0.0;
		DIMTYPE count=0;
		long N=0;
		CONSTRESULTSITER iter;
		for(iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); N+=iter->sub->get_d_raw();}
		
		for(DIMTYPE f=0;f<Ysiz/*max_features()*/;f++) {
			long fcnt=0;
			for(iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); if(iter->sub->selected_raw(f)) fcnt++;}
			if(fcnt>0) {
				result+=((RETURNTYPE)(fcnt)/(RETURNTYPE)(N)) * ((RETURNTYPE)(fcnt-1)/(RETURNTYPE)(size-1)); 
				count++;
			}
		}
		if(count==0) return 0.0;
		long Y=Ysiz;
		if(Y<1) Y=count;
		long D=N%Y, H=N%size;
		RETURNTYPE CWmin=(RETURNTYPE)((RETURNTYPE)N*(RETURNTYPE)N-(RETURNTYPE)Y*(RETURNTYPE)(N-D)-D*D)/(RETURNTYPE)((RETURNTYPE)Y*(RETURNTYPE)N*(RETURNTYPE)(size-1));
		RETURNTYPE CWmax=(RETURNTYPE)(H*H+N*(size-1)-H*size)/(RETURNTYPE)(N*(size-1));
		if(CWmax>CWmin) return (result-CWmin)/(CWmax-CWmin); else return result;
	} 
	else if(size==1) return 1.0;
	else return 0.0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stability_SH() const
{
	const long size=results.size();
	if(size>1)
	{
		RETURNTYPE result=0.0;
		DIMTYPE count=0;
		long fcntsum=0;
		CONSTRESULTSITER iter;
		for(iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); fcntsum+=iter->sub->get_d_raw();}
		
		for(DIMTYPE f=0;f<max_features();f++) {
			long fcnt=0;
			for(iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); if(iter->sub->selected_raw(f)) fcnt++;}
			if(fcnt>0) {
				result+=fcnt * (log((double)fcnt)/log((double)2.0)); 
				count++;
			}
		}
		if(count==0) return 0.0;
		return result/(fcntsum*(log((double)size)/log((double)2.0)));
	} 
	else if(size==1) return 1.0;
	else return 0.0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stability_PH(const DIMTYPE Ysiz) const
{
	assert(Ysiz>0);
	const long size=results.size();
	if(size>1)
	{
		RETURNTYPE result=0.0;
		for(DIMTYPE f=0;f<Ysiz;f++) {
			long fcnt=0;
			for(CONSTRESULTSITER iter=results.begin();iter!=results.end();iter++) {assert(iter->sub); if(iter->sub->selected_raw(f)) fcnt++;}
			result+=((fcnt>size-fcnt)?(RETURNTYPE)fcnt/(RETURNTYPE)size:(RETURNTYPE)(size-fcnt)/(RETURNTYPE)size);
		}
		return 2.0*result/(RETURNTYPE)Ysiz -1.0;
	} 
	else if(size==1) return 1.0;
	else return 0.0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::get_TI(const PSubset &sub1, const PSubset &sub2) const
{
	assert(sub1); 
	assert(sub2);
	const DIMTYPE Ysiz=maxf<DIMTYPE>(sub1->get_n_raw(),sub2->get_n_raw()); 
	static boost::scoped_array<int> TI_mask;
	static DIMTYPE TI_mask_size;
	assert(Ysiz>0);
	if(!TI_mask || TI_mask_size<Ysiz) {
		TI_mask.reset(new int[Ysiz]);
		TI_mask_size=Ysiz;
		notify("get_ATI reallocation.");
	}
	
	DIMTYPE f;
	for(f=0;f<Ysiz;f++) TI_mask[f]=0;
	// // pairwise intersection and union
	bool go=sub1->getFirstFeature(f);
	while(go)
	{
		assert(f>=0 && f<Ysiz);
		TI_mask[f]++;
		go=sub1->getNextFeature(f);
	}
	go=sub2->getFirstFeature(f);
	while(go)
	{
		assert(f>=0 && f<Ysiz);
		TI_mask[f]++;
		go=sub2->getNextFeature(f);
	}
	
	DIMTYPE SAS=0, SUS=0;
	for(f=0;f<Ysiz;f++) {if(TI_mask[f]==2) SAS++; if(TI_mask[f]!=0) SUS++;}
	if(SUS==0) {notify("Warning: getGenTani(): SUS==0."); return -1.0;}
	else return (RETURNTYPE)SAS/(RETURNTYPE)SUS;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stability_ATI() const
{
	const long size=results.size();
	if(size>1)
	{
		RETURNTYPE result=0.0;
		CONSTRESULTSITER iter1,iter2;
		for(iter1=results.begin();iter1!=results.end();iter1++) 
		{
			iter2=iter1; iter2++;
			while(iter2!=results.end()) {
				result+=get_TI(iter1->sub,iter2->sub);
				iter2++;
			}
		}
		return (2.0/(size*(size-1)))*result;
	} 
	else if(size==1) return 1.0;
	else return 0.0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::get_NHI(const DIMTYPE Ysiz, const PSubset &sub1, const PSubset &sub2) const
{
	assert(sub1); 
	assert(sub2);
	assert(Ysiz>0);
	static boost::scoped_array<int> NHI_mask;
	static DIMTYPE NHI_mask_size;
	assert(Ysiz>0);
	if(!NHI_mask || NHI_mask_size<Ysiz) {
		NHI_mask.reset(new int[Ysiz]);
		NHI_mask_size=Ysiz;
		notify("get_ANHI reallocation.");
	}
	
	DIMTYPE f;
	for(f=0;f<Ysiz;f++) NHI_mask[f]=0;
	// // pairwise intersection and union
	bool go=sub1->getFirstFeature(f);
	while(go)
	{
		assert(f>=0 && f<Ysiz);
		NHI_mask[f]++;
		go=sub1->getNextFeature(f);
	}
	go=sub2->getFirstFeature(f);
	while(go)
	{
		assert(f>=0 && f<Ysiz);
		NHI_mask[f]++;
		go=sub2->getNextFeature(f);
	}
	
	DIMTYPE SES=0;
	for(f=0;f<Ysiz;f++) if(NHI_mask[f]==1) SES++;
	return 1.0-(RETURNTYPE)SES/(RETURNTYPE)Ysiz;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stability_ANHI(const DIMTYPE Ysiz) const
{
	assert(Ysiz>0);
	const long size=results.size();
	if(size>1)
	{
		RETURNTYPE result=0.0;
		CONSTRESULTSITER iter1,iter2;
		for(iter1=results.begin();iter1!=results.end();iter1++) 
		{
			iter2=iter1; iter2++;
			while(iter2!=results.end()) {
				result+=get_NHI(Ysiz,iter1->sub,iter2->sub);
				iter2++;
			}
		}
		return (2.0/(size*(size-1)))*result;
	} 
	else if(size==1) return 1.0;
	else return 0.0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::similarity_IC(const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const
{
	const long size1=results.size();
	const long size2=rtse.results.size();
	if(size1>0 && size2>0)
	{
		const DIMTYPE Ysiz=maxf<DIMTYPE>(max_features(),rtse.max_features());
		assert(Ysiz>0);
		static boost::scoped_array<int> F1,F2;
		static DIMTYPE F_size;
		if(!F1 || !F2 || F_size<Ysiz) {
			F1.reset(new int[Ysiz]);
			F2.reset(new int[Ysiz]);
			F_size=Ysiz;
			notify("similarity_IC reallocation.");
		}
		DIMTYPE f;
		for(f=0;f<Ysiz;f++) F1[f]=F2[f]=0;
		for(f=0;f<Ysiz;f++) {
			CONSTRESULTSITER iter1;
			for(iter1=results.begin();iter1!=results.end();iter1++) {assert(iter1->sub); if(iter1->sub->selected_raw(f)) F1[f]++;}
			CONSTRESULTSITER iter2;
			for(iter2=rtse.results.begin();iter2!=rtse.results.end();iter2++) {assert(iter2->sub); if(iter2->sub->selected_raw(f)) F2[f]++;}
		}
		RETURNTYPE result=0.0;
		DIMTYPE count=0;
		for(f=0;f<Ysiz;f++) {
			if(F1[f]>0 || F2[f]>0) { 
				result+=fabs( (RETURNTYPE)(F1[f])/(RETURNTYPE)(size1) - (RETURNTYPE)(F2[f])/(RETURNTYPE)(size2) ); 
				count++;
			}
		}
		assert(count>0);
		return 1.0 - result/(RETURNTYPE)count;
	} else {notify("Warning: similarity_IC(): nothing to compare."); return 0.0;}
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::similarity_ICW(const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const
{
	const long size1=results.size();
	const long size2=rtse.results.size();
	if(size1>0 && size2>0)
	{
		const DIMTYPE Ysiz=maxf<DIMTYPE>(max_features(),rtse.max_features());
		assert(Ysiz>0);
		static boost::scoped_array<int> F1,F2;
		static DIMTYPE F_size;
		if(!F1 || !F2 || F_size<Ysiz) {
			F1.reset(new int[Ysiz]);
			F2.reset(new int[Ysiz]);
			F_size=Ysiz;
			notify("similarity_IC reallocation.");
		}
		DIMTYPE f;
		for(f=0;f<Ysiz;f++) F1[f]=F2[f]=0;
		for(f=0;f<Ysiz;f++) {
			CONSTRESULTSITER iter1;
			for(iter1=results.begin();iter1!=results.end();iter1++) {assert(iter1->sub); if(iter1->sub->selected_raw(f)) F1[f]++;}
			CONSTRESULTSITER iter2;
			for(iter2=rtse.results.begin();iter2!=rtse.results.end();iter2++) {assert(iter2->sub); if(iter2->sub->selected_raw(f)) F2[f]++;}
		}
		RETURNTYPE fcntsum=0.0;
		for(f=0;f<Ysiz;f++) {
			fcntsum+= maxf<RETURNTYPE>( (RETURNTYPE)(F1[f])/(RETURNTYPE)(size1) , (RETURNTYPE)(F2[f])/(RETURNTYPE)(size2) );// + fabs( (double)(F1[f])/(double)(sets1) - (double)(F2[f])/(double)(sets2) );
		}
		assert(fcntsum>0);
		RETURNTYPE result=0.0;
		for(f=0;f<Ysiz;f++) {
			result+=(maxf<RETURNTYPE>( (RETURNTYPE)(F1[f])/(RETURNTYPE)(size1) , (RETURNTYPE)(F2[f])/(RETURNTYPE)(size2) ) / fcntsum) * fabs( (RETURNTYPE)(F1[f])/(RETURNTYPE)(size1) - (RETURNTYPE)(F2[f])/(RETURNTYPE)(size2) ); 
		}
		return 1.0 - result;	
	} 
	else {notify("Warning: similarity_ICW(): nothing to compare."); return 0.0;}
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::similarity_IATI(const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const
{
	const long size1=results.size();
	const long size2=rtse.results.size();
	if(size1>0 && size2>0)
	{
		RETURNTYPE result=0.0;
		CONSTRESULTSITER iter1;
		for(iter1=results.begin();iter1!=results.end();iter1++) 
		{
			assert(iter1->sub);
			CONSTRESULTSITER iter2;
			for(iter2=rtse.results.begin();iter2!=rtse.results.end();iter2++) 
			{
				assert(iter2->sub);
				result+=get_TI(iter1->sub,iter2->sub);
			}
		}
		return (1.0/((RETURNTYPE)size1*(RETURNTYPE)(size2)))*result;	
	} 
	else {notify("Warning: similarity_IATI(): nothing to compare."); return 0.0;}
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
RETURNTYPE Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::similarity_IANHI(const DIMTYPE Ysiz, const Result_Tracker_Stability_Evaluator<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> &rtse) const
{
	const long size1=results.size();
	const long size2=rtse.results.size();
	if(size1>0 && size2>0)
	{
		assert(Ysiz>0);
		RETURNTYPE result=0.0;
		CONSTRESULTSITER iter1;
		for(iter1=results.begin();iter1!=results.end();iter1++) 
		{
			assert(iter1->sub);
			CONSTRESULTSITER iter2;
			for(iter2=rtse.results.begin();iter2!=rtse.results.end();iter2++) 
			{
				assert(iter2->sub);
				result+=get_NHI(Ysiz,iter1->sub,iter2->sub);
			}
		}
		return (1.0/((RETURNTYPE)size1*(RETURNTYPE)(size2)))*result;	
	} 
	else {notify("Warning: similarity_IANHI(): nothing to compare."); return 0.0;}
}

} // namespace
#endif // FSTRESULTTRACKERSTABILEVAL_H ///:~
