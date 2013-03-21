#ifndef FSTCLASSIFIERNORMALBAYES_H
#define FSTCLASSIFIERNORMALBAYES_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    classifier_normal_bayes.hpp
   \brief   Implements Bayes classifier based on normal (gaussian) model
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
#include <sstream>
#include <cmath>
#include <cstring> // memcpy
#include "error.hpp"
#include "global.hpp"
#include "classifier.hpp"
#include "model_normal.hpp"
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

/*! \brief Implements Bayes classifier based on normal (gaussian) model
 \note Classifier_Normal_Bayes should be able to handle _model learning
       in two ways: either [default] to learn() new subspace model in each train() - usable
       with FS involving multiple DataAccessor training splits, or to
       pre-learn() full space model and subsequently only narrow() the
       model in each train() - much faster, but usable only with non-changing
       set of training patterns, i.e., for one split data access only
*/
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Classifier_Normal_Bayes : public Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> { // abstract class
public:
	typedef boost::shared_ptr<DATAACCESSOR> PDataAccessor;
	typedef boost::shared_ptr<SUBSET> const PSubset;
	typedef typename DATAACCESSOR::PPattern PPattern;
	Classifier_Normal_Bayes();
	virtual ~Classifier_Normal_Bayes() {notify("Classifier_Normal_Bayes destructor.");}

	/*! pre-learning mode: 
	    if true, calls once _model->learn() for full set size and later in train() uses only narrow()ing to access submatrixes
	             (makes sense only as long as training data do not change, i.e., within one split)
	    if false (default), learns new model in each train() call 
	*/
	void enable_prelearn_mode(const PDataAccessor da);
	void disable_prelearn_mode() {_prelearn_mode=false;}
	bool get_prelearn_mode() const {return _prelearn_mode;}
	
	// NOTE: must! be called before train() or test() calls whenever the training set changes (i.e., with changing da Splits)
	void initialize(const PDataAccessor da); // must be called to pre-compute mean[] and cov[]

	virtual bool classify(DIMTYPE &cls, const PPattern &pattern);  // classifies pattern, returns the respective class index
	virtual bool train(const PDataAccessor da, const PSubset sub); // learns from designated training part of data
	virtual bool test(RETURNTYPE &result, const PDataAccessor da); // estimates accuracy using designated test data
	
	DIMTYPE get_n() const {assert(_model); return _model->get_n();}
	DIMTYPE get_d() const {assert(_model); return _model->get_d();}

	Classifier_Normal_Bayes* clone() const;
	Classifier_Normal_Bayes* sharing_clone() const {throw fst_error("Classifier_Normal_Bayes::sharing_clone() not supported, use Classifier_Normal_Bayes::clone() instead.");}
	Classifier_Normal_Bayes* stateless_clone() const {throw fst_error("Classifier_Normal_Bayes::stateless_clone() not supported, use Classifier_Normal_Bayes::clone() instead.");}
	
	virtual std::ostream& print(std::ostream& os) const {os << "Classifier_Normal_Bayes()"; return os;}
private:
	Classifier_Normal_Bayes(const Classifier_Normal_Bayes& cnb); // copy-constructor 
protected:
	boost::scoped_ptr<Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > _model;

	DIMTYPE _classes, _features; // size of arrays below
	boost::scoped_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _LUtemp; // to store matrix LU decomposition
	boost::scoped_array<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _inverse; // matrix for each class (lazy allocation)
	boost::scoped_array<REALTYPE> _det; // constant for each class (lazy allocation)
	boost::scoped_array<REALTYPE> _constant; // constant for each class (lazy allocation)
	boost::scoped_array<REALTYPE> _pxw; // p(x|class) (lazy allocation)
	boost::scoped_array<REALTYPE> _Pwx; // P(class|x) (lazy allocation)

private:
	bool _prelearn_mode;
	boost::scoped_array<DIMTYPE> _index;
	DIMTYPE _subfeatures;
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Classifier_Normal_Bayes(const Classifier_Normal_Bayes& cnb) :
	_classes(cnb._classes),
	_features(cnb._features),
	_prelearn_mode(cnb._prelearn_mode),
	_subfeatures(cnb._subfeatures)
{
	notify("Classifier_Normal_Bayes copy-constructor.");
	if(cnb._model) _model.reset(new Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(*cnb._model));
	if(cnb._LUtemp) _LUtemp.reset(new Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>(*cnb._LUtemp));
	if(_classes>0)
	{
		_inverse.reset(new Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>[_classes]);
		for(DIMTYPE c=0;c<_classes;c++) _inverse[c]=cnb._inverse[c];
		_det.reset(new REALTYPE[_classes]); memcpy((void *)_det.get(),(void *)(cnb._det).get(),sizeof(REALTYPE)*_classes);
		_constant.reset(new REALTYPE[_classes]); memcpy((void *)_constant.get(),(void *)(cnb._constant).get(),sizeof(REALTYPE)*_classes);
		_pxw.reset(new REALTYPE[_classes]); memcpy((void *)_pxw.get(),(void *)(cnb._pxw).get(),sizeof(REALTYPE)*_classes);
		_Pwx.reset(new REALTYPE[_classes]); memcpy((void *)_Pwx.get(),(void *)(cnb._Pwx).get(),sizeof(REALTYPE)*_classes);
	}
	_index.reset(new DIMTYPE[_features]); memcpy((void *)_index.get(),(void *)(cnb._index).get(),sizeof(DIMTYPE)*_features);
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>* Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::clone() const
{
	Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> *clone=new Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(*this);
	clone->set_cloned();
	return clone;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Classifier_Normal_Bayes()
{
	notify("Classifier_Normal_Bayes constructor.");
	_model.reset(new Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>);
	_classes=0;
	_features=0;
	_subfeatures=0;
	_prelearn_mode=false;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::enable_prelearn_mode(const PDataAccessor da)
{
	notify("Classifier_Normal_Bayes::init_prelearn_mode().");
	assert(_model);
	assert(da);
	assert(da->getNoOfFeatures()>0);
	assert(da->getNoOfClasses()>0);
	_model->learn(da);
	_prelearn_mode=true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(const PDataAccessor da)
{
	notify("Classifier_Normal_Bayes::initialize().");
	assert(da);
	assert(da->getNoOfFeatures()>0);
	assert(da->getNoOfClasses()>0);
	
	// (re)allocate buffers
	if(!_LUtemp) _LUtemp.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(da->getNoOfFeatures(),true));
	if(da->getNoOfFeatures()>_LUtemp->get_n_max()) _LUtemp->reset(da->getNoOfFeatures(),true);
	if(!_inverse || _classes!=da->getNoOfClasses()) _inverse.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>[da->getNoOfClasses()]);
	for(DIMTYPE c=0;c<da->getNoOfClasses();c++) if(da->getNoOfFeatures()>_inverse[c].get_n_max()) _inverse[c].reset(da->getNoOfFeatures());
	if(!_det || _classes!=da->getNoOfClasses()) _det.reset(new REALTYPE[da->getNoOfClasses()]);
	if(!_constant || _classes!=da->getNoOfClasses()) _constant.reset(new REALTYPE[da->getNoOfClasses()]);
	if(!_pxw || _classes!=da->getNoOfClasses()) _pxw.reset(new REALTYPE[da->getNoOfClasses()]);
	if(!_Pwx || _classes!=da->getNoOfClasses()) _Pwx.reset(new REALTYPE[da->getNoOfClasses()]);
	_classes=da->getNoOfClasses();

	if(!_index || _features!=da->getNoOfFeatures()) _index.reset(new DIMTYPE[da->getNoOfFeatures()]);
	_features=da->getNoOfFeatures();
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::train(const PDataAccessor da, const PSubset sub)
{
	// NOTE: mean[] and cov[] must be pre-computed using initialize().
	//       Explicit call is needed here to ensure wrapper functionality whenever training
	//       data change (split change). To ensure correct functionality when
	//       called from Criterion_Wrapper, _prelearn_mode must be se to false (this is default)
	
	notify("Classifier_Normal_Bayes::train().");
	assert(_model);
	assert(da);
	if(_prelearn_mode) assert(get_n()>0);
	assert(da->getNoOfFeatures()>0);
	assert(sub);
	assert(sub->get_frozen_mode()==false);
	if(_prelearn_mode) assert(sub->get_n_raw()==get_n());
	//{
	//	ostringstream sos; sos << "sub->get_n_raw()="<<sub->get_n_raw() << " da->getNoOfFeatures()="<<da->getNoOfFeatures() << endl;
	//	syncout::print(std::cout,sos);
	//}
	assert(sub->get_n_raw()==da->getNoOfFeatures());
	
	if(_prelearn_mode) _model->narrow_to(sub);
	else {_model->learn(da,sub); _model->denarrow();}

	// pre-compute matrix inverse, determinants depending on current subset
	REALTYPE tmpval=2.0*M_PI; for(DIMTYPE f=1;f<sub->get_d_raw();f++) tmpval*=2.0*M_PI;
	_LUtemp->redim(sub->get_d_raw());
	for(DIMTYPE c=0; c<_model->get_classes(); c++)
	{
		_inverse[c].redim(get_d());
		_model->get_cov()[c].LUdecompose(*_LUtemp);
		_model->get_cov()[c].invert(_inverse[c],*_LUtemp);
		//{
		//	ostringstream sos; sos << "_inverse[c="<<c<<"]:"<<endl<<_inverse[c]<<endl;
		//	syncout::print(std::cout,sos);
		//}
		_det[c]=_model->get_cov()[c].determinant(*_LUtemp);
		_constant[c]=sqrt(tmpval*_det[c]);
	}

	// prepare feature subset index buffering
	DIMTYPE f;
	bool b;
	_subfeatures=0;
	for(b=sub->getFirstFeature(f);b==true;b=sub->getNextFeature(f)) {assert(_subfeatures<get_n()); _index[_subfeatures++]=f;}
	
#ifdef DEBUG
	{
		ostringstream sos; sos << "index: "; for(f=0;f<_subfeatures;f++) sos << _index[f] << " "; sos << endl;
		syncout::print(std::cout,sos);
	}
#endif
	assert(_subfeatures==sub->get_d_raw());
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify(DIMTYPE &cls, const PPattern &pattern)
{
	// NOTE: mean[] and cov[] must be pre-computed using initialize()
	// NOTE: _inverse[] _det[] and _constant[] must be pre-computed
	notify("Classifier_Normal_Bayes::test().");
	assert(_model);
	assert(get_n()>0);
	assert(_model->get_classes()>1);
	assert(_classes==_model->get_classes());
	assert(_index);
	assert(_subfeatures==_model->get_d());
	assert(_subfeatures>0);
		
	DIMTYPE f1, f2;
	DIMTYPE c_cand;
	REALTYPE ecoef, px, tmpval;

	// compute p(x|w) and p(x)
	px=0.0;
	for(c_cand=0;c_cand<_classes;c_cand++)
	{
		ecoef=0.0;
		for(f1=0;f1<_subfeatures;f1++)
		{
			tmpval=0.0;
			for(f2=0;f2<_subfeatures;f2++)
			{
				tmpval+=pattern[_index[f2]]*_inverse[c_cand].at_raw(f2,f1);//[pROW+f1];
			}
			ecoef+=tmpval*pattern[_index[f1]];
		}
		_pxw[c_cand]=exp(-0.5*ecoef)/_constant[c_cand];
		px+=_pxw[c_cand]*_model->get_Pc(c_cand);
	}
	// compute P(w|x)
	for(c_cand=0;c_cand<_classes;c_cand++) _Pwx[c_cand]=(_pxw[c_cand]*_model->get_Pc(c_cand))/px;
	// find maximum P(w|x)
	tmpval=_Pwx[0]; cls=0;
	for(c_cand=1;c_cand<_classes;c_cand++) if(_Pwx[c_cand]>tmpval) {tmpval=_Pwx[c_cand]; cls=c_cand;}

	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Normal_Bayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test(RETURNTYPE &result, const PDataAccessor da)
{
	// NOTE: mean[] and cov[] must be pre-computed using initialize()
	// NOTE: _inverse[] _det[] and _constant[] must be pre-computed
	notify("Classifier_Normal_Bayes::test().");
	assert(_model);
	assert(get_n()>0);
	assert(_model->get_classes()>1);
	assert(_classes==_model->get_classes());
	assert(da);
	if(_prelearn_mode) assert(da->getNoOfFeatures()==get_n()); 
	else assert(da->getNoOfFeatures()>=get_n());
	assert(da->getNoOfClasses()>0);
	assert(_index);
	assert(_subfeatures==_model->get_d());
	assert(_subfeatures>0);
		
	typename DATAACCESSOR::PPattern p;
	IDXTYPE s,i;
	IDXTYPE count, correct;
	DIMTYPE clstmp;
	DIMTYPE _features=da->getNoOfFeatures();

	bool b;
	const DIMTYPE da_test_loop=1; // to avoid mixup of get*Block() loops of different types
	
	count=0;
	correct=0;
	for(DIMTYPE c_test=0;c_test<_classes;c_test++)
	{
		da->setClass(c_test);
		for(b=da->getFirstBlock(TEST,p,s,da_test_loop);b==true;b=da->getNextBlock(TEST,p,s,da_test_loop)) for(i=0;i<s;i++)
		{
			if(!classify(clstmp,&p[i*_features])) return false;
			if(clstmp==c_test) correct++;
			count++;
		}
	}
	assert(count>0);
	result=(RETURNTYPE)correct/(RETURNTYPE)count;
#ifdef DEBUG
	{
		ostringstream sos; sos << " result=" << result << endl;
		syncout::print(std::cout,sos);
	}
#endif	
	return true;
}


} // namespace
#endif // FSTCLASSIFIERNORMALBAYES_H ///:~
