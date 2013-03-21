#ifndef FSTMODELMULTINOM_H
#define FSTMODELMULTINOM_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    model_multinom.hpp
   \brief   Implements multinomial model
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
#include <limits>
#include <cmath>
#include <cstring> // memcpy
#include "error.hpp"
#include "global.hpp"
#include "model.hpp"
#include "indexed_vector.hpp"
#include "indexed_matrix.hpp"

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

//! Implements multinomial model
//
// \note only non-negative integers assumed as data
template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Model_Multinomial : public Model<SUBSET,DATAACCESSOR> {
public:
	typedef boost::shared_ptr<DATAACCESSOR> PDataAccessor;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Model_Multinomial() : _classes(0), _n_max(0), _n(0), _d(0), _allpatterns(0), _doc_avg_length(0) {notify("Model_Multinomial constructor");}
	Model_Multinomial(const Model_Multinomial& mm); // copy-constructor
	virtual ~Model_Multinomial() {notify("Model_Multinomial destructor");}

	virtual void learn(PDataAccessor da); // learn model of full dimensionality
	virtual void learn(PDataAccessor da, const PSubset sub); // learn model of lower dimensionality

	void narrow_to(const PSubset sub); // fill _index[] -> access to data is then mapped to sub-matrix
	void denarrow(); // reset _index[] -> access is reset to the full matrix

	void compute_theta();
	void compute_MI(); // Mutual Information of individual features
	void compute_IB(); // Individual Bhattacharyya of individual features

	DIMTYPE get_n_max() const {return _n_max;}
	DIMTYPE get_n() const {return _n;}
	DIMTYPE get_d() const {return _d;}
	DIMTYPE get_classes() const {return _classes;}
	
	REALTYPE get_doc_avg_length() const {return _doc_avg_length;}
	// NOTE: caller must ensure any access takes place only during _theta, _IB, _MI and _Pc existence
	REALTYPE get_Pc(const DIMTYPE c) const {assert(c>=0 && c<_classes); return _Pc[c];}
protected:
	template<class, class, class, class, class, class, class> friend class Criterion_Multinomial_Bhattacharyya;
	template<class, class, class, class, class, class, class> friend class Classifier_Multinomial_NaiveBayes;
	const boost::scoped_array<REALTYPE>& get_theta() const {return _theta;}
	const boost::scoped_array<REALTYPE>& get_IB() const {return _IB;}
	const boost::scoped_array<REALTYPE>& get_MI() const {return _MI;}
protected:
	DIMTYPE _classes; // initialized in learn(), WARNING: int sufficient ? note expressions like _classes*_n*_n
	// NOTE: matrix size is defined by:
	//       _n_max - hard constant representing maximum allocated space of size _n_max*_n_max
	//       _n     - "full feature set" dimensionality, _n<=_n_max. If Indexed_Matrix servers as output buffer
	//                to store matrices of various (reduced) dimensionalites, this is needed for correct functionality
	//                of at(), at_raw() etc. (both "raw" and "non-raw" matrix item access methods)
	//       _d     - "virtual subset" dimensionality, _d<=_n. Assuming the matrix holds _n*_n values (_n consecutively stored
	//                rows of _n values each), a virtual sub-matrix can be defined and accessed using at() ("non-raw" methods)
	DIMTYPE _n_max;
	DIMTYPE _n; // initialized in learn(), WARNING: int sufficient ? note expressions like _classes*_n*_n
	void compute_Nsuminclass(PDataAccessor da);
	boost::scoped_array<DATATYPE> _Nsuminclass;
	boost::scoped_array<REALTYPE> _theta;// actually P(term|class)
	boost::scoped_array<REALTYPE> _Pc;
	boost::scoped_array<REALTYPE> _Pc_d; // for compute_MI
	boost::scoped_array<REALTYPE> _Pv;
	boost::scoped_array<REALTYPE> _MI;
	boost::scoped_array<REALTYPE> _IB;

	DIMTYPE _d;
	// NOTE: _index defines a sub-space of subs-space defined in _learn_index
	boost::scoped_array<DIMTYPE> _index; //!< maps feature subset indexes to raw (full set) feature indexes, used for narrow()ing

	IDXTYPE _allpatterns; // initialized in learn()
	REALTYPE _doc_avg_length; // initialized in compute__theta(), needed in compute_IB()

	// NOTE: _learn_index here is not used for narrow()ing, but for learn(da,sub)
	boost::scoped_array<DIMTYPE> _learn_index; // maps virtual sub-space _n dimensions to original _n_max dimensions
};

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Model_Multinomial(const Model_Multinomial& mm) : // copy-constructor
	Model<SUBSET,DATAACCESSOR>(mm),
	_classes(mm._classes),
	_n_max(mm._n_max),
	_n(mm._n),
	_d(mm._d),
	_allpatterns(mm._allpatterns),
	_doc_avg_length(mm._doc_avg_length)
{
	notify("Model_Multinomial copy-constructor.");
	if(_classes>0) {
		_Pc.reset(new REALTYPE[_classes]); memcpy((void *)_Pc.get(),(void *)(mm._Pc).get(),sizeof(REALTYPE)*_classes);
		_Pc_d.reset(new REALTYPE[_classes]); memcpy((void *)_Pc_d.get(),(void *)(mm._Pc_d).get(),sizeof(REALTYPE)*_classes);
		if(_n_max>0) {
			_Nsuminclass.reset(new DATATYPE[_n_max*_classes]); memcpy((void *)_Nsuminclass.get(),(void *)(mm._Nsuminclass).get(),sizeof(DATATYPE)*_n_max*_classes);
			_theta.reset(new REALTYPE[_n_max*_classes]); memcpy((void *)_theta.get(),(void *)(mm._theta).get(),sizeof(REALTYPE)*_n_max*_classes);
		}
	}
	if(_n_max>0)
	{
		_Pv.reset(new REALTYPE[_n_max]); memcpy((void *)_Pv.get(),(void *)(mm._Pv).get(),sizeof(REALTYPE)*_n_max);
		_MI.reset(new REALTYPE[_n_max]); memcpy((void *)_MI.get(),(void *)(mm._MI).get(),sizeof(REALTYPE)*_n_max);
		_IB.reset(new REALTYPE[_n_max]); memcpy((void *)_IB.get(),(void *)(mm._IB).get(),sizeof(REALTYPE)*_n_max);
		// to store subsubspace info of the (possibly subspace-learned) learned model
		_index.reset(new DIMTYPE[_n_max]); memcpy((void *)_index.get(),(void *)(mm._index).get(),sizeof(DIMTYPE)*_n_max);
		// to store subspace info for the to-be-learned model
		_learn_index.reset(new DIMTYPE[_n_max]); memcpy((void *)_learn_index.get(),(void *)(mm._learn_index).get(),sizeof(DIMTYPE)*_n_max);
	}
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::learn(PDataAccessor da)
{
	assert(da);
	assert(da->getNoOfClasses()>0);
	assert(da->getNoOfFeatures()>0);
	PSubset fullset(new SUBSET(da->getNoOfFeatures()));
	fullset->select_all();
#ifdef DEBUG
	{std::ostringstream sos; sos << "fullset = "<< *fullset << std::endl; syncout::print(std::cout,sos);}
#endif
	learn(da,fullset);
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::learn(PDataAccessor da, const PSubset sub)
{
#ifdef DEBUG
	{std::ostringstream sos; sos <<"Model_Multinomial::learn():"<<std::endl; syncout::print(std::cout,sos);}
#endif
	assert(sub);
	assert(da->getNoOfClasses()>0);
	assert(da->getNoOfFeatures()>0);
	assert(sub->get_n_raw()==da->getNoOfFeatures());

	if(_classes!=da->getNoOfClasses() || sub->get_n_raw()>_n_max) // insufficient buffers - reset constants and reallocate
	{
		_classes=da->getNoOfClasses();
		_n_max=sub->get_n_raw();

		_learn_index.reset(new DIMTYPE[_n_max]); // to store subspace info about the to-be-learned model
		_index.reset(new DIMTYPE[_n_max]); for(DIMTYPE f=0;f<_n_max;f++) _index[f]=f; // to store subsubspace info of the (possibly subspace-learned) learned model
		_Nsuminclass.reset(new DATATYPE[_n_max*_classes]);
		_theta.reset(new REALTYPE[_n_max*_classes]);
		_Pv.reset(new REALTYPE[_n_max]);
		_MI.reset(new REALTYPE[_n_max]);
		_IB.reset(new REALTYPE[_n_max]);
		_Pc.reset(new REALTYPE[_classes]);
		_Pc_d.reset(new REALTYPE[_classes]);
		// NOTE: class priors are estimated from complete data available through da.
		//       Note that when only parts of data are processed (i.e., using data splits), the
		//       estimates below would possibly not precisely depict the relative frequency of documents
		//       in stratified class parts, esp. if class sizes are small
		IDXTYPE _classsizesum=da->getClassSizeSum();
		assert(_classsizesum>0);
		if(_classsizesum==0) throw FST::fst_error("Model_Multinomial::learn: zero _classsizesum");
		for(DIMTYPE c=0;c<_classes;c++) _Pc[c]=(REALTYPE)da->getClassSize(c)/(REALTYPE)_classsizesum;
	}
	_n=sub->get_d_raw(); // actual to-be-learned model dimensionality is to be the one defined by the subset defined in sub
	_d=_n; // access to learned model is initially not narrow()ed
	DIMTYPE f, fi=0;
	for(bool b=sub->getFirstFeature(f);b==true;b=sub->getNextFeature(f)) {_learn_index[fi++]=f;}
	assert(fi==_n);
	
#ifdef DEBUG
	{
		std::ostringstream sos; sos << "_Pc:"<<std::endl;
		for(DIMTYPE c=0;c<_classes;c++) sos << _Pc[c] << " "; sos << std::endl;
		syncout::print(std::cout,sos);
	}
#endif
	compute_Nsuminclass(da); // sub info passed in form of _learn_index
	
	DATATYPE total_sum_length=0;
	DIMTYPE wCV=0;
	for(DIMTYPE c=0;c<_classes;c++) {
		for(DIMTYPE f=0;f<_n;f++) total_sum_length+=_Nsuminclass[wCV+f];
		wCV+=_n;
	}
	if(total_sum_length==0) {
		for(DIMTYPE c=0;c<_classes;c++) _Pc_d[c]=1.0/(REALTYPE)_classes; // workaround to prevent division by zero below
	} else {
		wCV=0;
		for(DIMTYPE c=0;c<_classes;c++) {
			DATATYPE class_sum_length=0;
			for(DIMTYPE f=0;f<_n;f++) class_sum_length+=_Nsuminclass[wCV+f]; // in subset case this is done only for selected features
			_Pc_d[c]=(REALTYPE)class_sum_length/(REALTYPE)total_sum_length;
			wCV+=_n;
		}
	}
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::compute_Nsuminclass(PDataAccessor da)
{
	assert(_learn_index);
	assert(_Nsuminclass);
	for(DIMTYPE i=0;i<_classes*_n;i++) _Nsuminclass[i]=0;
	typename DATAACCESSOR::PPattern p;
	IDXTYPE cnt;
	assert(da->getSplitIndex()>0); // 
	DIMTYPE wCV=0;
	DIMTYPE _features=da->getNoOfFeatures();
	_allpatterns=0;
	for(DIMTYPE c=0;c<_classes;c++) {
		da->setClass(c);
		for(bool b=da->getFirstBlock(TRAIN,p,cnt);b!=false;b=da->getNextBlock(TRAIN,p,cnt)) {
			for(IDXTYPE i=0;i<cnt;i++) {
				for(DIMTYPE f=0;f<_n;f++) {_Nsuminclass[wCV+f]+=p[i*_features+_learn_index[f]];}
			}
			_allpatterns+=cnt;
		}
		wCV+=_n;
	}
#ifdef DEBUG
	{
		std::ostringstream sos; sos <<"_Nsuminclass:"<<std::endl;
		for(DIMTYPE c=0;c<_classes;c++) {for(DIMTYPE f=0;f<_n;f++) sos << _Nsuminclass[c*_n+f] << " "; sos << std::endl;}
		syncout::print(std::cout,sos);
	}
#endif
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::compute_theta()
{
	assert(_classes>0);
	assert(_n>0);
	assert(_d>0 || _d<=_n);
	assert(_index);
	assert(_allpatterns>0);
	assert(_Nsuminclass);
	assert(_theta);
	DATATYPE total_sum_length=0;
	DIMTYPE wCV=0, wCd=0;
	for(DIMTYPE c=0;c<_classes;c++) {
		DATATYPE class_sum_length=0;
		for(DIMTYPE f=0;f<_d;f++) class_sum_length+=_Nsuminclass[wCV+_index[f]]; // in subset case this is done only for selected features
		total_sum_length+=class_sum_length;
		for(DIMTYPE f=0;f<_d;f++) { // in subset case this is done only for selected features
			_theta[wCd++]=(1.0+(REALTYPE)_Nsuminclass[wCV+_index[f]])/((REALTYPE)_d+(REALTYPE)class_sum_length); // in subset case substitute d (subset size) for _n
		}
		wCV+=_n;
	}
	if(_allpatterns==0) throw FST::fst_error("Model_Multinomial::compute_theta: division by zero _allpatterns");
	_doc_avg_length=(REALTYPE)total_sum_length/(REALTYPE)_allpatterns;
#ifdef DEBUG
	{
		std::ostringstream sos; sos <<"_theta:"<<std::endl;
		for(DIMTYPE c=0;c<_classes;c++) {
			REALTYPE tmp=0.0;
			for(DIMTYPE f=0;f<_d;f++) {tmp+=_theta[c*_d+f]; sos << _theta[c*_d+f] << " "; sos << std::endl;}
			sos << "theta sum over class<"<<c<<">: "<<tmp<<std::endl;
		}
		sos << "total_sum_length="<<total_sum_length<<", doc_avg_length="<<_doc_avg_length<<std::endl;
		syncout::print(std::cout,sos);
	}
#endif
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::narrow_to(const PSubset sub)
{
	assert(_index);
	assert(sub);
	assert(sub->get_n_raw()==_n); // NOTE: sub should represent subspace of the learned subspace ! not of the full space
	assert(sub->get_d_raw()>0);
	DIMTYPE f;
	_d=0;
	for(bool b=sub->getFirstFeature(f);b!=false;b=sub->getNextFeature(f)) _index[_d++]=f;
	assert(_d>0 && _d<=_n);
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::denarrow()
{
	assert(_index);
	for(DIMTYPE i=0;i<_n;i++) _index[i]=i;
	_d=_n;
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::compute_MI()
{
	// NOTE: needs _theta computed for full feature set
	assert(_classes>0);
	assert(_n>0);
	assert(_Nsuminclass);
	assert(_theta);
	assert(_Pc);
	assert(_Pv);
	assert(_MI);
	for(DIMTYPE f=0;f<_n;f++) _MI[f]=0.0;
	DIMTYPE wCV=0;
	DATATYPE sum=0;
	for(DIMTYPE c=0;c<_classes;c++) {
		for(DIMTYPE f=0;f<_n;f++) {
			_MI[f]+=_Nsuminclass[wCV+f];
			sum+=_Nsuminclass[wCV+f];
		}
		wCV+=_n;
	}
	for(DIMTYPE f=0;f<_n;f++) {if(sum>0.0) _Pv[f]=(_MI[f])/((REALTYPE)sum); else _Pv[f]=0.0;}
	
	for(DIMTYPE f=0;f<_n;f++) {
		_MI[f]=0.0;
		if(_Pv[f]>0.0) {
			wCV=0;
			for(DIMTYPE c=0;c<_classes;c++) {
				//_MI[f]+=_theta[wCV+f]*_Pc_d[c]*fabs( (REALTYPE)log(_theta[wCV+f]/_Pv[f]) ); // IG abs
				//_MI[f]+=_theta[wCV+f]*_Pc_d[c]*(REALTYPE)log(_theta[wCV+f]/_Pv[f]); // IG
				//_MI[f]+=_Pc_d[c]*fabs( (REALTYPE)log(_theta[wCV+f]/_Pv[f]) ); // MI abs
				_MI[f]+=_Pc_d[c]*(REALTYPE)log(_theta[wCV+f]/_Pv[f]); // MI
				wCV+=_n;
			}
		}
	}
#ifdef DEBUG
	{
		std::ostringstream sos; sos <<"_MI:"<<std::endl;
		for(DIMTYPE f=0;f<_n;f++) sos << _MI[f] << " "; sos << std::endl;
		syncout::print(std::cout,sos);
	}
#endif
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::compute_IB()
{
	// NOTE: needs _theta computed for full feature set
	assert(_classes>0);
	assert(_n>0);
	assert(_doc_avg_length>0);
	assert(_theta);
	assert(_Pc);
	assert(_IB);
	for(DIMTYPE f=0;f<_n;f++) _IB[f]=0.0;
	DIMTYPE wCV1,wCV2;
	DIMTYPE combs;
	REALTYPE _thetasum, value;
	for(DIMTYPE f=0;f<_n;f++) {
		value=0.0; combs=0;
		wCV1=0;
		for(DIMTYPE c1=0;c1<_classes;c1++) {
			wCV2=wCV1+_n;
			for(DIMTYPE c2=c1+1;c2<_classes;c2++) {
				_thetasum=sqrt(_theta[wCV1+f]*_theta[wCV2+f])+sqrt((1.0-_theta[wCV1+f])*(1.0-_theta[wCV2+f]));
#ifdef DEBUG
				{std::ostringstream sos; sos << _thetasum<< std::endl; syncout::print(std::cout,sos);}
#endif
				value+=((-_doc_avg_length)*(REALTYPE)log(_thetasum))*_Pc_d[c1]*_Pc_d[c2];
				combs++;
				wCV2+=_n;
			}
			wCV1+=_n;
		}
		assert(combs>0);
		if(combs==0) throw FST::fst_error("Model_Multinomial::compute_IB: division by zero combs");
		_IB[f]=value/(REALTYPE)combs; 
	}
#ifdef DEBUG
	{
		std::ostringstream sos; sos <<"_IB:"<<std::endl;
		for(DIMTYPE f=0;f<_n;f++) sos << _IB[f] << " "; sos << std::endl;
		syncout::print(std::cout,sos);
	}
#endif
}


} // namespace
#endif // FSTMODELMULTINOM_H ///:~
