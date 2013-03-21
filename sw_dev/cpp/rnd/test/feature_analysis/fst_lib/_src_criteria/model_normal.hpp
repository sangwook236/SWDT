#ifndef FSTMODELNORMAL_H
#define FSTMODELNORMAL_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    model_normal.hpp
   \brief   Implements normal (gaussian) model
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at http://fst.utia.cz
   \date    October 2010
   \version 3.0.0.beta
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

//! Implements normal (gaussian) model
template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Model_Normal : public Model<SUBSET,DATAACCESSOR> {
public:
	typedef boost::shared_ptr<DATAACCESSOR> PDataAccessor;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Model_Normal() : _classes(0), _n_max(0), _n(0), _d(0) {notify("Model_Normal (empty) constructor");}
	Model_Normal(const Model_Normal& mn); // copy-constructor
	virtual ~Model_Normal() {notify("Model_Normal destructor");}
	
	virtual void learn(PDataAccessor da); // learn model of full dimensionality
	virtual void learn(PDataAccessor da, const PSubset sub); // learn model of lower dimensionality
	
	// NOTE: the learn() methods create model structures. narrow() methods allow access to sub-space in existing learned structures
	void narrow_to(const PSubset sub); // fill _index[] -> access to data is then mapped to sub-matrix
	void denarrow(); // reset _index[] -> access is reset to the full matrix
	
	DIMTYPE get_n_max() const {return _n_max;}
	DIMTYPE get_n() const {return _n;}
	DIMTYPE get_d() const {return _d;}
	DIMTYPE get_classes() const {return _classes;}
	
	// NOTE: caller must ensure any access takes place only during _mean, _Pc and _cov existence
	REALTYPE get_Pc(const DIMTYPE c) const {assert(c>=0 && c<_classes); return _Pc[c];}
protected:
	template<class, class, class, class, class, class, class> friend class Classifier_Normal_Bayes;
	template<class, class, class, class, class, class, class> friend class Criterion_Normal_Bhattacharyya;
	template<class, class, class, class, class, class, class> friend class Criterion_Normal_GMahalanobis;
	template<class, class, class, class, class, class, class> friend class Criterion_Normal_Divergence;
	const boost::scoped_array<Indexed_Vector<REALTYPE,DIMTYPE,SUBSET> >& get_mean() const {return _mean;}
	const boost::scoped_array<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> >& get_cov() const {return _cov;}
protected:
	DIMTYPE _classes; // WARNING: int sufficient ? note expressions like _classes*_n*_n
	// NOTE: matrix size is defined by:
	//       _n_max - hard constant representing maximum allocated space of size _n_max*_n_max
	//       _n     - "full feature set" dimensionality, _n<=_n_max. If Indexed_Matrix servers as output buffer
	//                to store matrices of various (reduced) dimensionalites, this is needed for correct functionality
	//                of at(), at_raw() etc. (both "raw" and "non-raw" matrix item access methods)
	//       _d     - "virtual subset" dimensionality, _d<=_n. Assuming the matrix holds _n*_n values (_n consecutively stored
	//                rows of _n values each), a virtual sub-matrix can be defined and accessed using at() ("non-raw" methods)
	DIMTYPE _n_max;
	DIMTYPE _n; // WARNING: int sufficient ? note expressions like _classes*_n*_n
	boost::scoped_array<REALTYPE> _Pc;
	boost::scoped_array<Indexed_Vector<REALTYPE,DIMTYPE,SUBSET> > _mean;
	boost::scoped_array<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _cov;
	// NOTE: all cov[] and mean[] implement redundantly the same subspace indexing logic. It would be
	// more effective to extract the "index" array to standalone object, one instance,
	// to be shared using boost::shared_ptr<> anywhere needed ?
	DIMTYPE _d;
	// NOTE: _n and _d should be always synchronized with those inside _mean and _cov;
	
	// NOTE: _learn_index here is not used for narrow()ing, but for learn(da,sub)
	boost::scoped_array<DIMTYPE> _learn_index; // maps virtual sub-space _n dimensions to original _n_max dimensions
};

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Model_Normal(const Model_Normal& mn) : // copy-constructor
	Model<SUBSET,DATAACCESSOR>(mn),
	_classes(mn._classes),
	_n_max(mn._n_max),
	_n(mn._n),
	_d(mn._d)
{
	notify("Model_Normal copy-constructor.");
	if(_classes>0) {
		_Pc.reset(new REALTYPE[_classes]); memcpy((void *)_Pc.get(),(void *)(mn._Pc).get(),sizeof(REALTYPE)*_classes);
		// yes I know the default constructor calls are superfluous
		_mean.reset(new Indexed_Vector<REALTYPE,DIMTYPE,SUBSET>[_classes]);
		_cov.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>[_classes]);
		for(DIMTYPE c=0;c<_classes;c++) {_mean[c]=mn._mean[c]; _cov[c]=mn._cov[c];}
	}
	if(_n_max>0)
	{
		// to store subspace info for the to-be-learned model
		_learn_index.reset(new DIMTYPE[_n_max]); memcpy((void *)_learn_index.get(),(void *)(mn._learn_index).get(),sizeof(DIMTYPE)*_n_max);
	}
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::learn(PDataAccessor da)
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
void Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::learn(PDataAccessor da, const PSubset sub)
{
	notify("Model_Normal::learn():");
	assert(da->getNoOfClasses()>0);
	assert(da->getNoOfFeatures()>0);
	assert(sub);
	assert(sub->get_n_raw()==da->getNoOfFeatures());
	assert(sub->get_d_raw()>0);

	if(_classes!=da->getNoOfClasses()) // || _n!=da->getNoOfFeatures())
	{
		_classes=da->getNoOfClasses();
		_mean.reset(new Indexed_Vector<REALTYPE,DIMTYPE,SUBSET>[_classes]);
		_cov.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>[_classes]);
		_Pc.reset(new REALTYPE[_classes]);
		_n_max=0; // request re-allocation in the following
	}
	// NOTE: class priors are inefficiently estimated in each learn() call even if da does not change
	// NOTE: class priors are estimated from complete data available through da.
	//       Note that when only parts of data are processed (i.e., using data splits), the
	//       estimates below would possibly not precisely depict the relative frequency of documents
	//       in stratified class parts, esp. if class sizes are small
	IDXTYPE _allpatterns=da->getClassSizeSum(); // just to estimate class priors
	assert(_allpatterns>0);
	for(DIMTYPE c=0;c<_classes;c++) _Pc[c]=(REALTYPE)da->getClassSize(c)/(REALTYPE)_allpatterns;

	if(sub->get_n_raw()>_n_max) // insufficient buffers - reset constants and reallocate
	{
		_n_max=sub->get_n_raw();
		for(DIMTYPE c=0;c<_classes;c++) {
			_mean[c].reset(_n_max);
			_cov[c].reset(_n_max);
		}
		_learn_index.reset(new DIMTYPE[_n_max]); // to store subspace info for the to-be-learned model
	}
	
	_n=sub->get_d_raw(); // actual to-be-learned model dimensionality is to be the one defined by the subset defined in sub
	for(DIMTYPE c=0;c<_classes;c++) {
		_mean[c].redim(_n); _mean[c].set_all_raw_to(0.0);
		_cov[c].redim(_n); _cov[c].set_all_raw_to(0.0);
	}
	_d=_n; // access to learned model is initially not narrow()ed
	DIMTYPE f, fi=0;
	for(bool b=sub->getFirstFeature(f);b==true;b=sub->getNextFeature(f)) {_learn_index[fi++]=f;}
	assert(fi==_n);
	
	typename DATAACCESSOR::PPattern p;
	IDXTYPE cnt;
	assert(da->getSplitIndex()>0); // splitting loop to be maintained externally
	// mean
	for(DIMTYPE c=0;c<_classes;c++) {
		da->setClass(c);
		assert(da->getNoOfPatterns(TRAIN)>0);
#ifdef DEBUG
		{std::ostringstream sos; sos << "da->getNoOfPatterns(TRAIN) [class="<<c<<"] = "<<da->getNoOfPatterns(TRAIN)<<std::endl; syncout::print(std::cout,sos);}
#endif
		for(bool b=da->getFirstBlock(TRAIN,p,cnt);b!=false;b=da->getNextBlock(TRAIN,p,cnt)) {
			for(IDXTYPE i=0;i<cnt;i++) 
				for(DIMTYPE f=0;f<_n;f++) (_mean[c])[f]+=p[i*_n_max+_learn_index[f]];
		}
		for(DIMTYPE f=0;f<_n;f++) (_mean[c])[f]/=da->getNoOfPatterns(TRAIN); 
	}
	
#ifdef DEBUG
	{
		std::ostringstream sos; sos << "_Pc:"<<std::endl;
		for(DIMTYPE c=0;c<_classes;c++) sos << _Pc[c] << " "; sos << std::endl;
		sos << "_mean:"<<std::endl;
		for(DIMTYPE c=0;c<_classes;c++) {for(DIMTYPE f=0;f<_n;f++) sos << (_mean[c])[f] << " "; sos << std::endl;}
		syncout::print(std::cout,sos);
	}
#endif
	// cov
	for(DIMTYPE c=0;c<_classes;c++) {
		da->setClass(c);
		for(bool b=da->getFirstBlock(TRAIN,p,cnt);b!=false;b=da->getNextBlock(TRAIN,p,cnt)) {
			for(IDXTYPE i=0;i<cnt;i++) {
				for(DIMTYPE f1=0;f1<_n;f1++)
				for(DIMTYPE f2=f1;f2<_n;f2++){
					_cov[c].at_raw(f1,f2)+=(p[i*_n_max+_learn_index[f1]]-(_mean[c])[f1])*(p[i*_n_max+_learn_index[f2]]-(_mean[c])[f2]);
				}
			}
		}
	}
	// normalize and copy to lower triangle
	for(DIMTYPE c=0;c<_classes;c++) {
		da->setClass(c);
		REALTYPE pats=da->getNoOfPatterns(TRAIN)-1;
		assert(pats>0);
		if(pats<1) pats=1;
		for(DIMTYPE f1=0;f1<_n;f1++)
			for(DIMTYPE f2=0;f2<f1;f2++) {
				_cov[c].at_raw(f1,f2)=_cov[c].at_raw(f2,f1);
			}
		for(DIMTYPE f1=0;f1<_n;f1++)
			for(DIMTYPE f2=0;f2<_n;f2++) {
				_cov[c].at_raw(f1,f2)/=pats;
			}
	}

	
#ifdef DEBUG
	{
		std::ostringstream sos; sos << "_cov:"<<std::endl;
		for(DIMTYPE c=0;c<_classes;c++) {for(DIMTYPE f1=0;f1<_n;f1++) {for(DIMTYPE f2=0;f2<_n;f2++) sos << _cov[c].at_raw(f1,f2) << " "; sos << std::endl;} sos << std::endl;}
		syncout::print(std::cout,sos);
	}
#endif
}


template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::narrow_to(const PSubset sub)
{
	assert(_mean);
	assert(_cov);
	assert(sub->get_n_raw()==_n); // NOTE: sub should represent subspace of the learned subspace ! not of the full space
	assert(sub->get_d_raw()>0);
	for(DIMTYPE c=0;c<_classes;c++) {
		_mean[c].narrow_to(sub);
		_cov[c].narrow_to(sub);
	}
	_d=sub->get_d_raw();
}

template<typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::denarrow()
{
	assert(_mean);
	assert(_cov);
	for(DIMTYPE c=0;c<_classes;c++) {
		_mean[c].denarrow();
		_cov[c].denarrow();
	}
	_d=_n;
}

} // namespace
#endif // FSTMODELNORMAL_H ///:~
