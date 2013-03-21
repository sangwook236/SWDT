#ifndef FSTCLASSIFIERKNN_H
#define FSTCLASSIFIERKNN_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    classifier_knn.hpp
   \brief   Implements k-Nearest Neighbor classifier
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
    of FST3, or if in doubt, to FST3 maintainer (see http://fst.utia.cz)
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
#include <cmath>
#include <list>
#include "classifier.hpp"
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

//! Implements k-Nearest Neighbor classifier
template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
class Classifier_kNN : public Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> { 
public:
	typedef Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent;
	typedef boost::shared_ptr<DATAACCESSOR> PDataAccessor;
	typedef boost::shared_ptr<SUBSET> PSubset;
	typedef typename DATAACCESSOR::PPattern PPattern;
	Classifier_kNN() {set_k(1); _nns_max_size=1; notify("Classifier_kNN constructor.");}
	virtual ~Classifier_kNN() {notify("Classifier_kNN destructor.");}

	void set_k(const DIMTYPE k) {assert(k>0); _k=k; _nns_max_size=_k;} // NOTE: _nns_max_size may be set larger to help avoiding ties
	DIMTYPE get_k() const {return _k;}

	virtual bool classify(DIMTYPE &cls, const PPattern &pattern);  // classifies pattern, returns the respective class index
	virtual bool train(const PDataAccessor da, const PSubset sub); // with kNN there is actually no training. This just stores pointer to training data to be later accessed from test()
	virtual bool test(RETURNTYPE &result, const PDataAccessor da); // estimates accuracy using designated test data
	
	Classifier_kNN* clone() const;
	Classifier_kNN* sharing_clone() const {throw fst_error("Classifier_kNN::sharing_clone() not supported, use Classifier_kNN::clone() instead.");}
	Classifier_kNN* stateless_clone() const {throw fst_error("Classifier_kNN::stateless_clone() not supported, use Classifier_kNN::clone() instead.");}
	
	virtual std::ostream& print(std::ostream& os) const;
private:
	Classifier_kNN(const Classifier_kNN& cknn); // copy-constructor 
protected:
	// NOTE: for given k it is better to keep ((k-1)*NoOfClasses+1) nearest neighbours to prevent ties (provided enough neighbours exist)
	// NOTE: equal neighbour distances to different classes are not handled extra
	
	//! Holds information on distance and data-class membership of neighbors processed in Classifier_kNN
	class Neighbour{
	public:
		Neighbour() {}
		Neighbour(const RETURNTYPE value, const DIMTYPE cls) {_value=value; _cls=cls;}
		Neighbour(const Neighbour& nei) {_value=nei._value; _cls=nei._cls;}	//!< copy constructor
		RETURNTYPE _value; 
		DIMTYPE _cls;
	};
	DIMTYPE _k;
	DIMTYPE _k_enough;
	DIMTYPE _nns_max_size;
	void sort_in(const RETURNTYPE value, const DIMTYPE cls);
	DIMTYPE get_most_freq_cls();
	typename std::list<Neighbour>::iterator iter;
	std::list<Neighbour> _nns; // implemented descending ... the closest neighbour is the last
	std::vector<DIMTYPE> cls_freqs;
	
	boost::scoped_ptr<DISTANCE> _distance;
	PDataAccessor _da_train; // with kNN there is actually no training. This is meant to store pointer to training data to be later accessed from test()
};

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::Classifier_kNN(const Classifier_kNN& cknn) :
	_k(cknn._k),
	_k_enough(cknn._k_enough),
	_nns_max_size(cknn._nns_max_size),
	_nns(cknn._nns),
	cls_freqs(cknn.cls_freqs)
{
	notify("Classifier_kNN constructor.");
	if(cknn._distance) _distance.reset(cknn._distance->clone());
}

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>* Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::clone() const
{
	Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE> *clone=new Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>(*this);
	clone->set_cloned();
	return clone;
}


template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
bool Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::train(const PDataAccessor da, const PSubset sub)
{
	assert(da);
	assert(sub);
	if(!_distance || _distance->get_n()<sub->get_n_raw()) _distance.reset(new DISTANCE(sub->get_n_raw()));
	_distance->narrow_to(sub);
	_da_train=da;
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
bool Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::classify(DIMTYPE &cls, const PPattern &pattern)
{
	// NOTE: uses da_train_loop=0 
	// NOTE: da and _da_train may point to the same instance -> be careful with setClass() etc.
	assert(_da_train);
	assert(_distance);
	assert(_distance->get_n()==_da_train->getNoOfFeatures());
	assert(_distance->get_d()>0);
	typename DATAACCESSOR::PPattern p2;
	IDXTYPE s2,i2;
	bool b;
	DIMTYPE _feats=_da_train->getNoOfFeatures();
	RETURNTYPE val;
	
	const DIMTYPE da_train_loop=0; // to avoid mixup of get*Block() loops of different types be careful when calling classify() from a different loop
	
	cls_freqs.resize(_da_train->getNoOfClasses());
	_nns_max_size=(_k-1)*_da_train->getNoOfClasses()+1; // to help avoiding ties
	_k_enough=DIMTYPE(floor(0.5*(RETURNTYPE)_k)+1); 
	
	_nns.clear();
	for(DIMTYPE c_train=0;c_train<_da_train->getNoOfClasses();c_train++)
	{
		_da_train->setClass(c_train);
		for(b=_da_train->getFirstBlock(TRAIN,p2,s2,da_train_loop);b==true;b=_da_train->getNextBlock(TRAIN,p2,s2,da_train_loop)) for(i2=0;i2<s2;i2++)
		{
			// p2[i2*_feats] is the beginning of the current pattern
			val=_distance->distance(pattern,&p2[i2*_feats]);
			sort_in(val,c_train);
		}
	}
	cls=get_most_freq_cls();

	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
bool Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::test(RETURNTYPE &result, const PDataAccessor da)
{
	// NOTE: da and _da_train may point to the same instance -> be careful with setClass() etc.
	assert(da);
	assert(_da_train);
	assert(_distance);
	assert(da->getNoOfFeatures()==_da_train->getNoOfFeatures());
	assert(_distance->get_n()==_da_train->getNoOfFeatures());
	typename DATAACCESSOR::PPattern p1;
	IDXTYPE s1,i1;
	bool b;
	IDXTYPE count=0, correct=0;
	DIMTYPE _feats=da->getNoOfFeatures();
	DIMTYPE clstmp;
	
	const DIMTYPE da_test_loop=1; // to avoid mixup of get*Block() loops of different types (must not! be 0, see classify())
	
	cls_freqs.resize(_da_train->getNoOfClasses());
	_nns_max_size=(_k-1)*_da_train->getNoOfClasses()+1; // to help avoiding ties
	_k_enough=DIMTYPE(floor(0.5*(RETURNTYPE)_k)+1); 
	
	for(DIMTYPE c_test=0;c_test<da->getNoOfClasses();c_test++)
	{
		da->setClass(c_test);
		for(b=da->getFirstBlock(TEST,p1,s1,da_test_loop);b==true;b=da->getNextBlock(TEST,p1,s1,da_test_loop)) {
			for(i1=0;i1<s1;i1++)
			{
				if(!classify(clstmp,&p1[i1*_feats])) return false; // \note classify() internally uses block loop index 0
				if(clstmp==c_test) ++correct;
				++count;
			}
			da->setClass(c_test); // necessary for the case of da pointing to the same object as _da_train
		}
	}
	if(count==0) return false;
	result = (RETURNTYPE)correct/(RETURNTYPE)count;
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
void Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::sort_in(const RETURNTYPE value, const DIMTYPE cls)
{
	iter=_nns.begin();
	while(iter!=_nns.end() && value < (*iter)._value) iter++;
	if(_nns.size()<_nns_max_size || iter!=_nns.begin()) {
		Neighbour tmp(value,cls);
		_nns.insert(iter,tmp);
		// NOTE: ? consider: if(as below && first value!= second value) _nns.pop_front() .. to prevent loosing tie info
		if(_nns.size()>_nns_max_size) _nns.pop_front();
	}
}

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
DIMTYPE Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::get_most_freq_cls()
{ // NOTE: destructs contents of _nns !
	assert(_da_train);
	DIMTYPE i,c,c_max,max=0;
	for(i=0;i<_da_train->getNoOfClasses();i++) cls_freqs[i]=0;
	while(_nns.size()>0) {
		c=(_nns.back()._cls);
		if(++cls_freqs[c]==_k_enough) return c;
		if(cls_freqs[c]>max) {max=cls_freqs[c]; c_max=c;}
		_nns.pop_back();
	}
	if(max>0) return c_max;
	else return _da_train->getNoOfClasses(); // wrong class
}

template<class RETURNTYPE, typename DATATYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR, class DISTANCE>
std::ostream& Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE>::print(std::ostream& os) const 
{
	os << "Clasifier_kNN(k=" << _k << ")";
	if(_distance) os << *_distance;
	return os;
}

} // namespace
#endif // FSTCLASSIFIERKNN_H ///:~
