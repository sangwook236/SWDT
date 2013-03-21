#ifndef FSTSUBSET_H
#define FSTSUBSET_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    subset.hpp
   \brief   Stores the info on currently selected features, enables generating permutations, etc.
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
#include <cstdlib> // rand()
#include "error.hpp"
#include "global.hpp"

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

template<class BINTYPE, typename DIMTYPE>
class Subset;

template<class BINTYPE, typename DIMTYPE>
std::ostream& operator<<(std::ostream& os, const Subset<BINTYPE,DIMTYPE>& tdb);

/*! \brief Stores the info on currently selected features, enables generating permutations, etc.
     \note the actual subset info is stored in _bin[],
     \note negative values mark de-selected features, positive values mark selected features. 0 is undefined.
     \note Basic identifier meaning: -1 feature deselected, 1 feature selected, 3 feature temporarily selected during traversal, 9 frozen feature (temporarily selected and protected from any changes)
*/
template<class BINTYPE, typename DIMTYPE>
class Subset {
	friend std::ostream& operator<< <>(std::ostream& os, const Subset<BINTYPE,DIMTYPE>& tdb);
public:
	Subset(const DIMTYPE n);	//!< n - max possible no. of features = allocation size
	Subset(const Subset<BINTYPE,DIMTYPE>& sub);	//!< copy constructor
	~Subset();
	void stateless_copy(Subset<BINTYPE,DIMTYPE>& sub); // copies subset from sub - all features converted to id_sel or id_desel only
	bool equivalent(const Subset<BINTYPE,DIMTYPE>& sub);	//!< tests whether sub represents the same subset (ignoring internal state)	

	// adaptable methods - output depends on _frozen_mode - to be called in SearchMethods()
	void select(const DIMTYPE i);	//!< mark i-th feature as selected
	void select(const DIMTYPE _from, const DIMTYPE _to) {assert(_to>_from); assert(_from>=0); assert(_to<_n); for(DIMTYPE i=_from;i<=_to;i++) select(i);}	//!< mark features _from,..,_to as selected
	void select_all() {for(DIMTYPE i=0;i<_n;i++) if(_bin[i]!=id_frozen) _bin[i]=id_sel;};	//!< mark all features as deselected
	void deselect(const DIMTYPE i);	//!< mark i-th feature as deselected
	void deselect(const DIMTYPE _from, const DIMTYPE _to) {assert(_to>_from); assert(_from>=0); assert(_to<_n); for(DIMTYPE i=_from;i<=_to;i++) deselect(i);}	//!< mark features _from,..,_to as deselected
	void deselect_all() {for(DIMTYPE i=0;i<_n;i++) if(_bin[i]!=id_frozen) _bin[i]=id_desel;};	//!< mark all features as deselected
	DIMTYPE get_n() const {DIMTYPE z=0; for(DIMTYPE i=0;i<_n;i++) if(_bin[i]==id_frozen) ++z; return _n-z;} 
	DIMTYPE get_d() const {DIMTYPE d=0; for(DIMTYPE i=0;i<_n;i++) if(_bin[i]>0 && _bin[i]!=id_frozen && _bin[i]!=-id_frozen) ++d; return d;};

	// "raw" versions - to be called in Criterion()
	void select_raw(const DIMTYPE i) {assert(i>=0&&i<_n); _bin[i]=id_sel;}	//!< mark i-th feature as selected
	void select_all_raw() {for(DIMTYPE i=0;i<_n;i++) _bin[i]=id_sel;};	//!< mark all features as deselected
	void deselect_raw(const DIMTYPE i) {assert(i>=0&&i<_n); _bin[i]=id_desel;}	//!< mark i-th feature as deselected
	void deselect_all_raw() {for(DIMTYPE i=0;i<_n;i++) _bin[i]=id_desel;};	//!< mark all features as deselected
	DIMTYPE get_n_raw() const {return _n;}
	DIMTYPE get_d_raw() const {DIMTYPE d=0; for(DIMTYPE i=0;i<_n;i++) if(_bin[i]>0) ++d; return d;};
	bool selected_raw(const DIMTYPE i) const {if(i>=0&&i<_n) return (_bin[i]>0); else return false;}//{assert(i>=0&&i<_n); return (_bin[i]>0);}

	void make_random_subset(const DIMTYPE d); // generate random subset of d features
	void make_random_subset(const DIMTYPE lo_d, const DIMTYPE hi_d); // random subset size will be from <lo_d,hi_d>
	void make_random_subset(const float d_prob); // d_prob must be from (0,1]; generates random subset of random size where the mean size over a series of calls is d_prob*no_of_all_features

	// NOTE: forward mode is treated independently in "unfrozen" and "frozen" modes
	void set_forward_mode(const bool forward=true);	//!< enables inverting the meaning of id_sel, id_desel etc.
	bool get_forward_mode() const {return _forward_mode;}
	
	// NOTE: turns on/off the frozen mode, i.e., fixes the currently selected features (id_sel) and ignores them transparently
	// during candidate subset traversals and selections/deselections etc. However, frozen features are
	// included in get*Feature() traversals
	// NOTE: get_d_raw()-get_d()= no. of currently frozen features
	void set_frozen_mode(const bool freeze=false);
	bool get_frozen_mode() const {return _frozen_mode;}
	
	// NOTE: regardles _frozen_mode always returns all selected features (id_sel, id_frozen, id_traverse)
	bool getFirstFeature(DIMTYPE &feat, bool selected=true, const int looplevel=0); // NOTE: selected and looplevel so far ignored
	bool getNextFeature(DIMTYPE &feat);

	bool getFirstCandidateSubset(const DIMTYPE g, bool reverse=false); // traversing through candidate subsets (generates g-touples of features)
	bool getNextCandidateSubset();
	DIMTYPE getNoOfCandidateSubsets(const DIMTYPE g, bool reverse=false); // returns loop size (the number of to-be-traversed subsets) - NOTE: no overflow checking implemented, better use with g close to 1 only
	bool getFirstTemporaryFeature(DIMTYPE &feat); // traversing through temporary features in current CandidateSubset
	bool getNextTemporaryFeature(DIMTYPE &feat); // traversing through temporary features in current CandidateSubset
	DIMTYPE getNoOfTemporaryFeatures(); // returns the number of temporarily selected features (indicated by tmp_id_traverse) in current CandidateSubset

private:
	const DIMTYPE _n;	//!< max. No. of features (full set size)
	const boost::scoped_array<BINTYPE> _bin;	//!< default format to store subset info
		
	BINTYPE id_sel;
	BINTYPE id_desel;
	BINTYPE id_traverse;
	BINTYPE id_frozen;
	BINTYPE tmp_id_sel; // local to get*CandidateSubset()
	BINTYPE tmp_id_desel; // local to get*CandidateSubset()
	BINTYPE tmp_id_traverse; // local to get*CandidateSubset()
	DIMTYPE get_combinations(const DIMTYPE n, const DIMTYPE k); // NOTE: no overflow checking
	
	bool _forward_mode;
	bool _frozen_mode;

	DIMTYPE _gen_features; // get*CandidateSubset()
	DIMTYPE _enum_temporary; // get*TemporaryFeature()

	DIMTYPE _enum_selected; // get*Feature()
};

template<class BINTYPE, typename DIMTYPE>
Subset<BINTYPE,DIMTYPE>::Subset(const DIMTYPE n) : _n(n), _bin(new BINTYPE[n]) 
{
	id_desel=-1; id_sel=1; id_traverse=3; id_frozen=9;
	for(DIMTYPE i=0;i<_n;i++) _bin[i]=id_desel;
	_forward_mode=true;
	_frozen_mode=false;
	_gen_features=0;
	_enum_selected=0;
	notify("Subset constructor.");
}


template<class BINTYPE, typename DIMTYPE>
Subset<BINTYPE,DIMTYPE>::~Subset() 
{
	notify("Subset destructor.");
}


template<class BINTYPE, typename DIMTYPE>
Subset<BINTYPE,DIMTYPE>::Subset(const Subset<BINTYPE,DIMTYPE>& sub) : _n(sub._n), _bin(new BINTYPE[sub._n])
{
	id_sel=sub.id_sel; id_desel=sub.id_desel; id_traverse=sub.id_traverse; id_frozen=sub.id_frozen;
	for(DIMTYPE i=0;i<_n;i++) _bin[i]=sub._bin[i];
	_forward_mode=sub._forward_mode;
	_frozen_mode=sub._frozen_mode;
	_gen_features=sub._gen_features;
	_enum_selected=sub._enum_selected;
	notify("Subset copy-constructor.");
}


template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::stateless_copy(Subset<BINTYPE,DIMTYPE>& sub)
{
	// WARNING: functionality differs depending on _forward_mode
	assert(_bin);
	assert(_n>=sub.get_n_raw());
	for(DIMTYPE i=0;i<_n;i++) if(_bin[i]!=id_frozen && _bin[i]!=-id_frozen) _bin[i]=-1; // NOTE: hard constant
	DIMTYPE f;
	for(bool b=sub.getFirstFeature(f);b!=false;b=sub.getNextFeature(f)) {
		assert(f>=0 && f<_n);
		if(_bin[f]!=id_frozen && _bin[f]!=-id_frozen) _bin[f]=1; // NOTE: hard constant
	}
}

template<class BINTYPE, typename DIMTYPE>
bool Subset<BINTYPE,DIMTYPE>::equivalent(const Subset<BINTYPE,DIMTYPE>& sub)
{
	assert(_bin);
	assert(_n>0);
	bool eq=true;
	DIMTYPE fidx=0;
	while(eq && fidx<_n) if((_bin[fidx]<=0 && sub._bin[fidx]>0)||(_bin[fidx]>0 && sub._bin[fidx]<=0)) eq=false; else ++fidx;
	return eq;
}

template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::select(const DIMTYPE i)
{
	assert(_bin);
	assert(i>=0&&i<get_n());
	DIMTYPE u=0, c=i;
	while(u<_n && _bin[u]==id_frozen) ++u;
	while(u<_n && c>0) {
		++u; --c;
		while(u<_n && _bin[u]==id_frozen) ++u;
	}
	assert(u<_n);
	_bin[u]=id_sel;
}


template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::deselect(const DIMTYPE i)
{
	assert(_bin);
	assert(i>=0&&i<get_n());
	DIMTYPE u=0, c=i;
	while(u<_n && _bin[u]==id_frozen) ++u;
	while(u<_n && c>0) {
		++u; --c;
		while(u<_n && _bin[u]==id_frozen) ++u;
	}
	assert(u<_n);
	_bin[u]=id_desel;
}

template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::make_random_subset(const DIMTYPE d)
{
	assert(_bin);
	assert(d>=1 && d<=get_n());
	// assert bin must contain >=d non-frozen slots
	DIMTYPE i,piv;
	for(i=0;i<_n;i++) if(_bin[i]!=id_frozen) _bin[i]=id_desel;
	for(i=0;i<d;i++)
	{ 
		piv=(DIMTYPE)(rand()%_n); assert(0<=piv && piv<_n);
		while(_bin[piv]!=id_desel) {piv++; if(piv>_n-1) piv=0;}
		_bin[piv]=id_sel;
	}
}

template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::make_random_subset(const DIMTYPE lo_d, const DIMTYPE hi_d)
{
	assert(lo_d>=1 && lo_d<=get_n());
	assert(hi_d>=1 && hi_d<=get_n());
	assert(lo_d<hi_d);
	DIMTYPE d;
	d=lo_d+(DIMTYPE)(rand()%(int)(hi_d-lo_d+1)); assert(lo_d<=d && d<=hi_d);
	make_random_subset(d);
}

template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::make_random_subset(const float d_prob) // d_prob must be from (0,1]; generates random subset of random size where the mean size over a series of calls is d_prob*no_of_all_features
{
	assert(_bin);
	assert(d_prob>0.0 && d_prob<=1.0);
	do {
		DIMTYPE i;
		for(i=0;i<_n;i++) if(_bin[i]!=id_frozen) {
			if((float)rand()/(float)RAND_MAX<d_prob) _bin[i]=id_sel;
			else _bin[i]=id_desel;
		}
	} while(get_d()==0);
}

template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::set_forward_mode(const bool forward)
{
	assert(_gen_features==0); // no traversal is going on
	assert(_enum_selected==0); // no feature iteration is going on
	assert(_frozen_mode==false); // in frozen mode change direction of traversal using getFirstCandidateSubset() 'reverse' parameter
	if(forward) {
		id_desel=-1; id_sel=1; id_traverse=3; /*if(_frozen_mode==0)*/ id_frozen=9;
		_forward_mode=true;
	} else {
		id_desel=1; id_sel=-1; id_traverse=-3; /*if(_frozen_mode==0)*/ id_frozen=-9;
		_forward_mode=false;
	}
}


template<class BINTYPE, typename DIMTYPE>
void Subset<BINTYPE,DIMTYPE>::set_frozen_mode(const bool freeze)
{
	assert(_bin);
	assert(_gen_features==0); // no traversal is going on
	assert(_enum_selected==0); // no feature iteration is going on
	assert(freeze!=_frozen_mode); // repeated initiation of the same mode would produce unexpected results
	if(freeze) {
		for(DIMTYPE i=0;i<_n;i++) if(_bin[i]==id_sel || _bin[i]==id_traverse) _bin[i]=id_frozen;
		_frozen_mode=true;
	} else {
		for(DIMTYPE i=0;i<_n;i++) if(_bin[i]==id_frozen || _bin[i]==id_traverse) _bin[i]=id_sel;
		_frozen_mode=false;
	}
}

template<class BINTYPE, typename DIMTYPE>
bool Subset<BINTYPE,DIMTYPE>::getFirstFeature(DIMTYPE &feat, bool selected, const int looplevel)
{
	assert(_bin);
	assert(_n>0);
	_enum_selected=0;
	while(_bin[_enum_selected]<=0 && _enum_selected<_n) ++_enum_selected;
	feat=_enum_selected;
	_enum_selected++;
	if(feat<_n) return true; else return false;
}


template<class BINTYPE, typename DIMTYPE>
bool Subset<BINTYPE,DIMTYPE>::getNextFeature(DIMTYPE &feat)
{
	assert(_bin);
	assert(_n>0);
	assert(_enum_selected>0);	// i.e., getFirstFeature has been called before
	while(_bin[_enum_selected]<=0 && _enum_selected<_n) ++_enum_selected;
	feat=_enum_selected;
	_enum_selected++;
	if(feat<_n) return true; else {_enum_selected=0; return false;}
}


template<class BINTYPE, typename DIMTYPE>
bool Subset<BINTYPE,DIMTYPE>::getFirstCandidateSubset(const DIMTYPE g, bool reverse) // traversing through candidate subsets (adding/removing) g-touples of features
{
	assert(_bin);
	assert(g>=1 && g<=_n);
	if(reverse) {tmp_id_sel=-id_sel; tmp_id_desel=-id_desel; tmp_id_traverse=-id_traverse;} 
	else {tmp_id_sel=id_sel; tmp_id_desel=id_desel; tmp_id_traverse=id_traverse;}	
	DIMTYPE rr=g;
	for(DIMTYPE i=0;((rr>0)&&(i<_n));i++) if(_bin[i]==tmp_id_desel) {_bin[i]=tmp_id_traverse; --rr;} /* initialize _bin for exhaustive step */
	if(rr==0) {_gen_features=g; return true;}
	else {for(DIMTYPE i=0;((rr>0)&&(i<_n));i++) if(_bin[i]==tmp_id_traverse) _bin[i]=tmp_id_desel; _gen_features=0; return false;}
}

template<class BINTYPE, typename DIMTYPE>
bool Subset<BINTYPE,DIMTYPE>::getNextCandidateSubset()
{
	assert(_bin);
	assert(_gen_features>0 && _gen_features<_n);
	DIMTYPE beg, piv, pom;
	/* finding new candidate configuration */
	for(beg=0;beg<_n && _bin[beg]!=tmp_id_traverse;beg++);
	for(piv=beg;piv<_n && _bin[piv]!=tmp_id_desel;piv++);
	if(piv==_n) { // no more configurations
		for(beg=0;beg<_n;beg++) if(_bin[beg]==tmp_id_traverse) _bin[beg]=tmp_id_desel;
		_gen_features=0;
		return false;
	}
	else
	{
		pom=piv; /* remember the position of first (desel) on the right */
		do piv--; while(_bin[piv]!=tmp_id_traverse); /* find the real pivot */
		_bin[piv]=tmp_id_desel; _bin[pom]=tmp_id_traverse; /* shift pivot to the right */
		pom=0;
		/* run "pom" from left, "piv" from right. the desel,travers pairs found are changed to travers,desel */
		if(piv>0) --piv;
		while((piv>0)&&(_bin[piv]!=tmp_id_traverse)) --piv;
		while((pom<piv)&&(_bin[pom]!=tmp_id_desel)) pom++;
		while(piv>pom)
		{
			_bin[piv]=tmp_id_desel; _bin[pom]=tmp_id_traverse;
			if(piv>0) --piv;
			while((piv>0)&&(_bin[piv]!=tmp_id_traverse)) --piv;
			while((pom<piv)&&(_bin[pom]!=tmp_id_desel))pom++;
		}
	}
	return true;
}

template<class BINTYPE, typename DIMTYPE>
DIMTYPE Subset<BINTYPE,DIMTYPE>::getNoOfCandidateSubsets(const DIMTYPE g, bool reverse) // returns loop size (the number of to-be-traversed subsets)
{
	// NOTE: no overflow checking implemented.
	//       this is no issue for g==1, but may become a problem with increasing g
	assert(_bin);
	assert(g>=1 && g<_n);
	BINTYPE id_empty; if(reverse) id_empty=-id_desel; else id_empty=id_desel;
	DIMTYPE ecnt=0; for(DIMTYPE i=0;i<_n;i++) if(_bin[i]==id_empty) ++ecnt;
	if(g>ecnt) return 0;
	if(g==1) return ecnt; // just acceleration
	else return get_combinations(ecnt,g);
}

template<class BINTYPE, typename DIMTYPE>
bool Subset<BINTYPE,DIMTYPE>::getFirstTemporaryFeature(DIMTYPE &feat) // traversing through temporary features in current CandidateSubset
{
	assert(_bin);
	assert(_n>0);
	_enum_temporary=0;
	while(_bin[_enum_temporary]!=tmp_id_traverse && _enum_temporary<_n) ++_enum_temporary;
	feat=_enum_temporary;
	_enum_temporary++;
	if(feat<_n) return true; else {_enum_temporary=0; return false;}
}

template<class BINTYPE, typename DIMTYPE>
bool Subset<BINTYPE,DIMTYPE>::getNextTemporaryFeature(DIMTYPE &feat) // traversing through temporary features in current CandidateSubset
{
	assert(_bin);
	assert(_n>0);
	assert(_enum_temporary>0);	// i.e., getFirstTemporaryFeature has been called before
	while(_bin[_enum_temporary]!=tmp_id_traverse && _enum_temporary<_n) ++_enum_temporary;
	feat=_enum_temporary;
	_enum_temporary++;
	if(feat<_n) return true; else {_enum_selected=0; return false;}
}
template<class BINTYPE, typename DIMTYPE>
DIMTYPE Subset<BINTYPE,DIMTYPE>::getNoOfTemporaryFeatures() // returns the number of temporarily selected features (indicated by tmp_id_traverse) in current CandidateSubset
{
	assert(_bin);
	assert(_n>0);
	DIMTYPE _sum=0;
	for(DIMTYPE i=0; i<_n; i++) if(_bin[i]==tmp_id_traverse) _sum++;
	return _sum;
}

template<class BINTYPE, typename DIMTYPE>
DIMTYPE Subset<BINTYPE,DIMTYPE>::get_combinations(const DIMTYPE n, const DIMTYPE k)
{
	assert(n>0);
	assert(k>=0);
	assert(n>=k);
	if(k==0 || n==k) return 1;
	unsigned long nfrac=(unsigned long)n, kfrac=(unsigned long)k, nsubkfrac=(unsigned long)n-k, tempcheck;
	for(unsigned long i=(unsigned long)n-1;i>1;i--) {
		tempcheck=nfrac;
		nfrac*=i;
		if(tempcheck>=nfrac) throw FST::fst_error("Subset::get_combinations() numeric representation overflow.");
	}
	for(unsigned long i=(unsigned long)k-1;i>1;i--) kfrac*=i;
	for(unsigned long i=(unsigned long)n-k-1;i>1;i--) nsubkfrac*=i;
	
	unsigned long ulresult=nfrac/(kfrac*nsubkfrac);
	DIMTYPE result=(DIMTYPE)ulresult;
	if(ulresult!=(unsigned long)result) throw FST::fst_error("Subset::get_combinations() numeric representation overflow.");
	return result;
}

template<class BINTYPE, typename DIMTYPE>
std::ostream& operator<<(std::ostream& os, const Subset<BINTYPE,DIMTYPE>& tdb)
{
	assert(tdb._bin);
	assert(tdb._n>0);
	assert(tdb._bin);
	const DIMTYPE d_raw=tdb.get_d_raw();
	os << "Subset<n=" << tdb._n << ", d=" << d_raw <<">: ";
#ifdef DEBUG
	for(DIMTYPE i=0;i<tdb._n;i++) {os << (int)tdb._bin[i] << " ";} 
	os << "  ~  ";
#endif
	if(d_raw==0) os << "empty";
	else for(DIMTYPE i=0;i<tdb._n;i++) {if(tdb._bin[i]>0) os << i << " ";}
	return os;
}

} // namespace
#endif // FSTSUBSET_H ///:~
