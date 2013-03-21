#ifndef FSTCLASSIFIERMULTINOMNAIVEBAYES_H
#define FSTCLASSIFIERMULTINOMNAIVEBAYES_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    classifier_multinom_naivebayes.hpp
   \brief   Implements Naive-like Bayes classifier based on multinomial model
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
#include "model_multinom.hpp"

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

/*! \brief Implements Naive-like Bayes classifier based on multinomial model */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Classifier_Multinomial_NaiveBayes : public Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> { // abstract class
	// \note In this case the classifier should be better called Naive-like Bayes than Naive Bayes
public:
	typedef boost::shared_ptr<DATAACCESSOR> PDataAccessor;
	typedef boost::shared_ptr<SUBSET> const PSubset;
	typedef typename DATAACCESSOR::PPattern PPattern;
	Classifier_Multinomial_NaiveBayes();
	virtual ~Classifier_Multinomial_NaiveBayes() {notify("Classifier_Multinomial_NaiveBayes destructor.");}

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

	Classifier_Multinomial_NaiveBayes* clone() const;
	Classifier_Multinomial_NaiveBayes* sharing_clone() const {throw fst_error("Classifier_Multinomial_NaiveBayes::sharing_clone() not supported, use Classifier_Multinomial_NaiveBayes::clone() instead.");}
	Classifier_Multinomial_NaiveBayes* stateless_clone() const {throw fst_error("Classifier_Multinomial_NaiveBayes::stateless_clone() not supported, use Classifier_Multinomial_NaiveBayes::clone() instead.");}
	
	virtual std::ostream& print(std::ostream& os) const {os << "Classifier_Multinomial_NaiveBayes()"; return os;}
private:
	Classifier_Multinomial_NaiveBayes(const Classifier_Multinomial_NaiveBayes& cmnb); // copy-constructor 
protected:
	boost::scoped_ptr<Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > _model;

	DIMTYPE _classes, _features; // size of arrays below
	boost::scoped_array<REALTYPE> _Pcd; // P(class|document) (lazy allocation)

private:
	bool _prelearn_mode;
	boost::scoped_array<DIMTYPE> _index;
	DIMTYPE _subfeatures;
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Classifier_Multinomial_NaiveBayes(const Classifier_Multinomial_NaiveBayes& cmnb) :
	_classes(cmnb._classes),
	_features(cmnb._features),
	_prelearn_mode(cmnb._prelearn_mode),
	_subfeatures(cmnb._subfeatures)
{
	notify("Classifier_Multinomial_NaiveBayes copy-constructor.");
	if(cmnb._model) _model.reset(new Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(*cmnb._model));
	if(_classes>0)
	{
		_Pcd.reset(new REALTYPE[_classes]); memcpy((void *)_Pcd.get(),(void *)(cmnb._Pcd).get(),sizeof(REALTYPE)*_classes);
	}
	_index.reset(new DIMTYPE[_features]); memcpy((void *)_index.get(),(void *)(cmnb._index).get(),sizeof(DIMTYPE)*_features);
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>* Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::clone() const
{
	Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> *clone=new Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(*this);
	clone->set_cloned();
	return clone;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Classifier_Multinomial_NaiveBayes()
{
	notify("Classifier_Multinomial_NaiveBayes constructor.");
	_model.reset(new Model_Multinomial<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>);
	_classes=0;
	_features=0;
	_subfeatures=0;
	_prelearn_mode=false;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::enable_prelearn_mode(const PDataAccessor da)
{
	notify("Classifier_Multinomial_NaiveBayes::init_prelearn_mode().");
	assert(_model);
	assert(da);
	assert(da->getNoOfFeatures()>0);
	assert(da->getNoOfClasses()>0);
	_model->learn(da);
	_prelearn_mode=true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(PDataAccessor da)
{
	notify("Classifier_Multinomial_NaiveBayes::initialize().");
	//assert(_model);
	assert(da);
	assert(da->getNoOfFeatures()>0);
	assert(da->getNoOfClasses()>0);
	
	if(!_Pcd || _classes!=da->getNoOfClasses()) _Pcd.reset(new REALTYPE[da->getNoOfClasses()]);
	_classes=da->getNoOfClasses();

	if(!_index || _features!=da->getNoOfFeatures()) _index.reset(new DIMTYPE[da->getNoOfFeatures()]);
	_features=da->getNoOfFeatures();
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::train(const PDataAccessor da, const PSubset sub)
{
	// NOTE: mean[] and cov[] must be pre-computed using initialize()
	// NOTE: explicit call is needed here to ensure wrapper functionality
	//       but a work-aroung can be implemented to optionally disable the call
	//       (would make sense when testing different subsets for the same da split)
	initialize(da);
	
	notify("Classifier_Multinomial_NaiveBayes::train().");
	assert(_model);
	assert(da);
	if(_prelearn_mode) assert(get_n()>0);
	assert(da->getNoOfFeatures()>0);
	assert(sub);
	assert(sub->get_frozen_mode()==false);
	if(_prelearn_mode) assert(sub->get_n_raw()==get_n());
	//{
	//	ostringstream sos;
	//	sos << "sub->get_n_raw()="<<sub->get_n_raw() << " da->getNoOfFeatures()="<<da->getNoOfFeatures() << std::endl;
	//	syncout::print(std::cout,sos);
	//}
	assert(sub->get_n_raw()==da->getNoOfFeatures());
	
	if(_prelearn_mode) _model->narrow_to(sub); // assume _n-dimensional model is pre-learned and needs to be narrow()ed only (NOTE: training data must not change since)
	else {_model->learn(da,sub); _model->denarrow();} // re-learn from scratch, training data has changed (i.e., after switch to next data split)

//	if(sub->get_d_raw()==1) // ??? single feature subsets can not be used for multinomial model based classification
//	{
//	}

	// prepare feature subset index buffering
	DIMTYPE f;
	bool b;
	for(b=sub->getFirstFeature(f),_subfeatures=0;b==true;b=sub->getNextFeature(f),_subfeatures++) {assert(_subfeatures<get_n()); _index[_subfeatures]=f;}
	
	_model->compute_theta(); //thetas to be used in classify() and test()
	
#ifdef DEBUG
	{ostringstream sos; sos << "index: "; for(f=0;f<_subfeatures;f++) sos << _index[f] << " "; syncout::print(std::cout,sos);}
#endif
	assert(_subfeatures==sub->get_d_raw());
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify(DIMTYPE &cls, const PPattern &pattern)
{
	// NOTE: mean[] and cov[] must be pre-computed using initialize()
	// NOTE: _inverse[] _det[] and _constant[] must be pre-computed
	notify("Classifier_Multinomial_NaiveBayes::test().");
	assert(_model);
	if(_prelearn_mode) assert(get_n()>0);
	assert(_model->get_classes()>1);
	assert(_classes==_model->get_classes());
	assert(_index);
	assert(_subfeatures==_model->get_d());
	if(!_prelearn_mode) assert(_subfeatures==_model->get_n());
	assert(_subfeatures>0);
		
	DIMTYPE f;
	DIMTYPE c_cand;
	REALTYPE res;
	DIMTYPE wTH;
	REALTYPE *theta=&(_model->get_theta()[0]); // dirty, but this object owns _model thus no memory corruption is possible
	REALTYPE *theta_tmp;

	if(_subfeatures==1) { 
		// NOTE: single features are unusable for classification because theta=1 in each class,
		//       thus log(theta)=0 and consequently Pcd[] does not depend on feature frequency
		cls=0; // consider all single features equally unusable
		return false;
	} else {
		// compute P(c|pattern) for each class
		wTH=0;
		for(c_cand=0;c_cand<_classes;c_cand++)
		{
			theta_tmp=&theta[wTH];
			res=0.0;
			for(f=0;f<_subfeatures;f++)
			{
				//{
				//	ostringstream sos; sos << "f:" << _index[f] << " ptmp[]=" << (REALTYPE)ptmp[_index[f]] << ", theta=" << _model->get_theta()[wTH+f] << ", log=" << log(_model->get_theta()[wTH+f]) << std::endl;
				//	syncout::print(std::cout,sos);
				//}
				res+=(REALTYPE)pattern[_index[f]] * log(theta_tmp[f]);
			}
			_Pcd[c_cand]=log(_model->get_Pc(c_cand))+res;
			//{
			//	ostringstream sos; sos << " Pc["<<c_cand<<"]=" << _model->get_Pc(c_cand) << ", log(Pc)=" << log(_model->get_Pc(c_cand)) << ", res="<< res << ", _Pcd[]=" << _Pcd[c_cand] << std::endl << std::endl;
			//	syncout::print(std::cout,sos);
			//}
			wTH+=_subfeatures;
		}
		// find maximum P[c|pattern]
		res=_Pcd[0]; cls=0;
		for(c_cand=1;c_cand<_classes;c_cand++) if(_Pcd[c_cand]>res) {res=_Pcd[c_cand]; cls=c_cand;}
	}
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_Multinomial_NaiveBayes<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test(RETURNTYPE &result, const PDataAccessor da)
{
	// NOTE: mean[] and cov[] must be pre-computed using initialize()
	// NOTE: _inverse[] _det[] and _constant[] must be pre-computed
	notify("Classifier_Multinomial_NaiveBayes::test().");
	assert(_model);
	if(_prelearn_mode) assert(get_n()>0);
	assert(_model->get_classes()>1);
	assert(_classes==_model->get_classes());
	assert(da);
	if(_prelearn_mode) assert(da->getNoOfFeatures()==get_n());
	assert(da->getNoOfClasses()>0);
	assert(da->getNoOfClasses()==_classes);
	assert(_index);
	assert(_subfeatures==_model->get_d());
	if(!_prelearn_mode) assert(_subfeatures==_model->get_n());
	assert(_subfeatures>0);
		
	typename DATAACCESSOR::PPattern p;
	IDXTYPE s,i;
	IDXTYPE count, correct;
	DIMTYPE _features=da->getNoOfFeatures();
	DIMTYPE clstmp;

	if(_subfeatures==1) { 
		// NOTE: single features are unusable for classification because theta=1 in each class,
		//       thus log(theta)=0 and consequently Pcd[] does not depend on feature frequency
		correct=0; count=1; // consider all single features equally unusable
	} else {
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
	}
	assert(count>0);
	result=(RETURNTYPE)correct/(RETURNTYPE)count;
#ifdef DEBUG
	{
		ostringstream sos; sos << " result=" << result << std::endl;
		syncout::print(std::cout,sos);
	}
#endif	
	return true;
}

} // namespace
#endif // FSTCLASSIFIERMULTINOMNAIVEBAYES_H ///:~
