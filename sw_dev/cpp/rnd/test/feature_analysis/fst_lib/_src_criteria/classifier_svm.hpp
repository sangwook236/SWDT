#ifndef FSTCLASSIFIERSVM_H
#define FSTCLASSIFIERSVM_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    classifier_svm.hpp
   \brief   Wraps external Support Vector Machine implementation (in LibSVM) to serve as FST3 classifier
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
#include <list>
#include <cstring> // memcpy
#include "error.hpp"
#include "global.hpp"
#include "classifier.hpp"
#include "svm.h"

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

// NOTE: this is the proxy to LIBSVM external library
//       therefore some FST coding conventions can not
//       be strictly kept

/*! \brief Wraps external Support Vector Machine implementation (in LibSVM) to serve as FST3 classifier
    \note clone() implementation assumes LibSVM's svm_model and svm_node is equivalent to that in LibSVM version 300

 \warning LIBSVM especially with LINEAR kernel seems to have occassional problems with certain C values on certain datasets and may freeze. 
          (Other kernels are more stable but not completely immune to this problem.) This is a problem outside FST3.
*/

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Classifier_LIBSVM : public Classifier<RETURNTYPE,DIMTYPE,SUBSET,DATAACCESSOR> { 
public:
	typedef boost::shared_ptr<DATAACCESSOR> PDataAccessor;
	typedef boost::shared_ptr<SUBSET> const PSubset;
	typedef typename DATAACCESSOR::PPattern PPattern;
	Classifier_LIBSVM();
	virtual ~Classifier_LIBSVM() {notify("Classifier_LIBSVM destructor."); cleanup();}

	void initialize(const PDataAccessor da); // must be called to pre-allocate data transfer buffers
protected:
	void allocate(); // to be called in initialize(), pre-allocates everything
	void cleanup(); // deallocates everything
public:
	void set_parameter_C(double newC) {parameters.C = newC;};
	void set_parameter_gamma(double newgamma) {parameters.gamma = newgamma;};
	void set_parameter_coef0(double newcoef0) {parameters.coef0 = newcoef0;};
	void set_kernel_type(int kernel_type) {parameters.kernel_type = kernel_type;}; // see svm.h for admissible values
	double get_parameter_C() const {return parameters.C;}
	double get_parameter_gamma() const {return parameters.gamma;}
	double get_parameter_coef0() const {return parameters.coef0;}
	int get_kernel_type() const {return parameters.kernel_type;}; // see svm.h for admissible values

	virtual bool classify(DIMTYPE &cls, const PPattern &pattern);  // classifies pattern, returns the respective class index
	virtual bool train(const PDataAccessor da, const PSubset sub); // learns from designated training part of data
	virtual bool test(RETURNTYPE &result, const PDataAccessor da); // estimates accuracy using designated test data
	
	bool optimize_parameters(const PDataAccessor da, const PSubset sub, const int max_points=100, const int max_throws=100, const double lgC_min=-5, const double lgC_max=9, const double lggamma_min=-15, const double lggamma_max=3, const double lgcoef0_min=-2, const double lgcoef0_max=5, std::ostream& os=std::cout);
	
	Classifier_LIBSVM* clone() const {return stateless_clone();} //! \note dirty workaround to enable easy usage in sequential_step ...
	Classifier_LIBSVM* sharing_clone() const {throw fst_error("Classifier_LIBSVM::sharing_clone() not supported, use Classifier_LIBSVM::stateless_clone() instead.");}
	Classifier_LIBSVM* stateless_clone() const;
	
	virtual std::ostream& print(std::ostream& os) const {os << "Classifier_LIBSVM(kernel="<<get_kernel_type()<<")"; return os;}
private:
	Classifier_LIBSVM(const Classifier_LIBSVM& csvm, int); // weak copy-constructor, does not copy LibSVM internal structures
protected:
	// the following to be initialized using initialize() based on dataaccessor info
	IDXTYPE _all_patterns;
	DIMTYPE _classes;
	DIMTYPE _features;
	struct svm_problem problem; // data to be stored in here
	struct svm_parameter parameters;
	struct svm_model *model;
	struct svm_node *onepattern;

	bool svm_class_weighing;

	//! Nested class to hold parameter candidates in the course of optimize_parameters() run
	class ParameterSet {
	public:
		ParameterSet(): _crit_value(0.0), _Cpar(1.0), _gamma(1.0), _coef0(1.0) {}
		ParameterSet(const RETURNTYPE crit_value, const double C, const double gamma, const double coef0) : _crit_value(crit_value), _Cpar(C), _gamma(gamma), _coef0(coef0) {}
		ParameterSet(const ParameterSet& ps) : _crit_value(ps._crit_value), _Cpar(ps._Cpar), _gamma(ps._gamma), _coef0(ps._coef0) {}
		void operator=(const ParameterSet& ps) {_crit_value=ps._crit_value; _Cpar=ps._Cpar; _gamma=ps._gamma; _coef0=ps._coef0;}
		RETURNTYPE _crit_value;
		double _Cpar; // _C is reserved
		double _gamma; 
		double _coef0;
	};
	typedef list<ParameterSet> PARAMSETLIST;
	typename PARAMSETLIST::iterator iter; 

private:
	boost::scoped_array<DIMTYPE> _index;
	DIMTYPE _subfeatures;	
};

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Classifier_LIBSVM(const Classifier_LIBSVM& csvm, int) : // copy-constructor
	_all_patterns(csvm._all_patterns),
	_classes(csvm._classes),
	_features(csvm._features),
	svm_class_weighing(csvm.svm_class_weighing),
	_subfeatures(csvm._subfeatures)
{
	notify("Classifier_LIBSVM (weak) copy-constructor.");
	// does not copy LibSVM-defined structures, except for 'parameters'
	// other-than 'parameters' LibSVM structures are only pre-allocated using allocate()
	parameters.svm_type=csvm.parameters.svm_type;
	parameters.kernel_type=csvm.parameters.kernel_type;
	parameters.degree=csvm.parameters.degree;
	parameters.gamma=csvm.parameters.gamma;
	parameters.coef0=csvm.parameters.coef0;
	parameters.cache_size=csvm.parameters.cache_size;
	parameters.eps=csvm.parameters.eps;
	parameters.C=csvm.parameters.C;
	parameters.nr_weight=csvm.parameters.nr_weight;
	parameters.nu=csvm.parameters.nu;
	parameters.p=csvm.parameters.p;
	parameters.shrinking=csvm.parameters.shrinking;
	parameters.probability=csvm.parameters.probability;	
	allocate();
	model=NULL;
	if(csvm.parameters.weight_label) memcpy((void *)parameters.weight_label,(void *)csvm.parameters.weight_label,(long)(_classes) * sizeof(int));
	if(csvm.parameters.weight) memcpy((void *)parameters.weight,(void *)csvm.parameters.weight,(long)(_classes) * sizeof(double));
	if(csvm._index) {_index.reset(new DIMTYPE[_features]); memcpy((void *)_index.get(),(void *)(csvm._index).get(),sizeof(DIMTYPE)*_features);}
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>* Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::stateless_clone() const
{
	Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> *clone=new Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(*this,(int)0);
	clone->set_cloned();
	return clone;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Classifier_LIBSVM()
{
	notify("Classifier_LIBSVM constructor.");
	_all_patterns=0;
	_classes=0;
	_features=0;
	problem.x=NULL;
	problem.y=NULL;
	onepattern=NULL;
	parameters.weight_label=NULL;
	parameters.weight=NULL;
	model=NULL;
	_subfeatures=0;

	// default parameters
	parameters.svm_type=C_SVC;
	parameters.kernel_type=RBF; //LINEAR;
	parameters.degree=2;
	parameters.gamma=/*standard_gamma=*/1.0;
	parameters.coef0=1.0;
	parameters.cache_size=10;
	parameters.eps=0.001;
	parameters.C=/*standard_C=*/1.0;
	parameters.nr_weight=0;
	parameters.weight_label=NULL; // just sharing a pointer
	parameters.weight=NULL; // just sharing a pointer
	parameters.nu=1.0;
	parameters.p=1.0;
	parameters.shrinking=0;
	parameters.probability=0;
	svm_class_weighing=false;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::optimize_parameters(const PDataAccessor da, const PSubset sub, const int max_points, const int max_throws, const double lgC_min, const double lgC_max, const double lggamma_min, const double lggamma_max, const double lgcoef0_min, const double lgcoef0_max, std::ostream& os)
{
	// \note: assumes Classifier_LIBSVM::initialize() had been called for this da
	// assumes initial C, gamma, and coef0 have been set using set_parameter_*(), or that default values are OK
	// (the initial values are used as search starting point)
	/*! \warning LIBSVM LINEAR kernel seems to have problems with certain C values on certain datasets. If the optimization
	             process seemingly freezes, try narrower lgC_min and lgC_max. Especially the lower bound seems
	             important to be increased.
	*/
	assert(da);
	assert(sub);
	assert(sub->get_d()>0);
	assert(max_points>0);
	assert(max_throws>0);
	assert(lgC_max>lgC_min);
	assert(lggamma_max>lggamma_min);
	assert(lgcoef0_max>lgcoef0_min);
	const double lgCrange=lgC_max-lgC_min;
	const double lggammarange=lggamma_max-lggamma_min;
	const double lgcoef0range=lgcoef0_max-lgcoef0_min;
	PARAMSETLIST _paramlist; // _paramlist.clear();
	ParameterSet tmpset, bestset;
	// evaluate the initial (current) parameter set
	RETURNTYPE val, maxval, firstval;
	RETURNTYPE result=0.0;
	RETURNTYPE cnt=0.0;
	for(bool b=da->getFirstSplit();b==true;b=da->getNextSplit()) {
		if(!train(da,sub)) return false;
		if(!test(val,da)) return false;
		result+=val;
		cnt+=1.0;
	}
	firstval=maxval=val=result/(RETURNTYPE)cnt;
	{
		ostringstream sos;
		sos << "SVM before optimization: accuracy=" << val << ", C=" << get_parameter_C();
		switch(get_kernel_type()) {
			case LINEAR:  sos << std::endl; break;
			case RBF:     sos << ", gamma=" << get_parameter_gamma() << std::endl; break;
			case POLY:    
			case SIGMOID: sos << ", gamma=" << get_parameter_gamma() << ", coef0=" << get_parameter_coef0() << std::endl; break;
		}
		syncout::print(os,sos);
	}
	_paramlist.push_back(ParameterSet(val,get_parameter_C(),get_parameter_gamma(),get_parameter_coef0()));
	for(int i=0;i<max_points;i++)
	{ // find and evaluate next parameter set candidate
		double maxdistmin=0.0;
		for(int j=0;j<max_throws;j++)
		{ // find next parameter set candidate
			tmpset._Cpar =exp(lgC_min+(((double)rand()*(double)(lgC_max-lgC_min))/(double)RAND_MAX));
			tmpset._gamma=exp(lggamma_min+(((double)rand()*(double)(lggamma_max-lggamma_min))/(double)RAND_MAX));
			tmpset._coef0=exp(lgcoef0_min+(((double)rand()*(double)(lgcoef0_max-lgcoef0_min))/(double)RAND_MAX));
			
			double distmin=-1.0; // potential new candidate (weighted) distance to already evaluated candidates
			for(iter=_paramlist.begin();iter!=_paramlist.end();iter++)
			{
				double dist=0.0;
				assert(get_kernel_type()==LINEAR || get_kernel_type()==POLY || get_kernel_type()==RBF || get_kernel_type()==SIGMOID);
				switch(get_kernel_type()) {
					case LINEAR:  dist=sqrt(((log(tmpset._Cpar)-log(iter->_Cpar))/lgCrange)*((log(tmpset._Cpar)-log(iter->_Cpar))/lgCrange)); break;
					case RBF:     dist=sqrt(((log(tmpset._Cpar)-log(iter->_Cpar))/lgCrange)*((log(tmpset._Cpar)-log(iter->_Cpar))/lgCrange)+((log(tmpset._gamma)-log(iter->_gamma))/lggammarange)*((log(tmpset._gamma)-log(iter->_gamma))/lggammarange)); break;
					case POLY:    
					case SIGMOID: dist=sqrt(((log(tmpset._Cpar)-log(iter->_Cpar))/lgCrange)*((log(tmpset._Cpar)-log(iter->_Cpar))/lgCrange)+((log(tmpset._gamma)-log(iter->_gamma))/lggammarange)*((log(tmpset._gamma)-log(iter->_gamma))/lggammarange)+((log(tmpset._coef0)-log(iter->_coef0))/lgcoef0range)*((log(tmpset._coef0)-log(iter->_coef0))/lgcoef0range)); break;
				}
				// candidates more distant from all others are preferred
				// distances are weighed to penalize closeness to worse-performing candidates
				assert(iter->_crit_value>0.0);
				if(distmin<0 || dist*iter->_crit_value<distmin) distmin=dist*iter->_crit_value;
			}
			assert(distmin>=0.0);
			if(distmin>maxdistmin)
			{ // better (more distant from those already tested, esp. from the bad ones) parameter set candidate found
				maxdistmin=distmin;
				bestset=tmpset;
			}
		}
		// evaluate the chosen candidate and add it to list
		set_parameter_C(bestset._Cpar);
		set_parameter_gamma(bestset._gamma);
		set_parameter_coef0(bestset._coef0);
		assert(svm_check_parameter(&problem,&parameters)==NULL);
		result=0.0;
		cnt=0.0;
		for(bool b=da->getFirstSplit();b==true;b=da->getNextSplit()) {
			if(!train(da,sub)) return false;
			if(!test(val,da)) return false;
			result+=val;
			cnt+=1.0;
		}
		val=result/(RETURNTYPE)cnt;
		bestset._crit_value=val;
		_paramlist.push_back(bestset);
		if(true)
		{
			ostringstream sos;
			sos << i+1 << ". ";
			if(val>maxval) {maxval=val; sos << "BEST accuracy=";} else sos << "test accuracy=";
			sos << bestset._crit_value << ", C=" << get_parameter_C();
			switch(get_kernel_type()) {
				case LINEAR:  sos << std::endl; break;
				case RBF:     sos << ", gamma=" << get_parameter_gamma() << std::endl; break;
				case POLY:    
				case SIGMOID: sos << ", gamma=" << get_parameter_gamma() << ", coef0=" << get_parameter_coef0() << std::endl; break;
			}
			syncout::print(os,sos);
		}
	}
	// identify the best parameter set among all tested ones
	assert(!_paramlist.empty());
	iter=_paramlist.begin();
	bestset=(*iter);
	while(iter!=_paramlist.end())
	{
		if(iter->_crit_value>bestset._crit_value) bestset=(*iter);
		iter++;
	}
	set_parameter_C(bestset._Cpar);
	set_parameter_gamma(bestset._gamma);
	set_parameter_coef0(bestset._coef0);
	{
		ostringstream sos;
		sos << "SVM after optimization: accuracy=" << bestset._crit_value << ", C=" << get_parameter_C();
		switch(get_kernel_type()) {
			case LINEAR:  sos << std::endl; break;
			case RBF:     sos << ", gamma=" << get_parameter_gamma() << std::endl; break;
			case POLY:    
			case SIGMOID: sos << ", gamma=" << get_parameter_gamma() << ", coef0=" << get_parameter_coef0() << std::endl; break;
		}
		sos << "Accuracy difference=" << bestset._crit_value-firstval << std::endl << std::endl;
		syncout::print(os,sos);
	}
	return false;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::allocate()
{
	try {
		//cleanup(); - not to be called here or it would destroy original data in copy-constructor
		assert(_all_patterns>0);
		assert(_classes>1);
		assert(_features>0);
		
		problem.y =               (double *)          malloc(_all_patterns * sizeof(double));
		problem.x =               (struct svm_node **)malloc((long)(_all_patterns) * sizeof(struct svm_node *));
		parameters.weight_label = (int *)             malloc((long)(_classes) * sizeof(int));
		parameters.weight =       (double *)          malloc((long)(_classes) * sizeof(double));
		onepattern =              (struct svm_node *) malloc((long)(_features+1) * sizeof(struct svm_node));
		for(IDXTYPE p=0;p<_all_patterns;p++) problem.x[p]=NULL;
		// pre-allocate all problem.x[] arrays to max size, to prevent re-allocation speed efficiency problems 
		// - WARNING! this leads to uneconomical use of memory
		for(IDXTYPE p=0;p<_all_patterns;p++) 
			problem.x[p] = (struct svm_node *) malloc((long)(_features+1) * sizeof(struct svm_node));

		// class penalty weighing disabled
		parameters.nr_weight=0;
		problem.l = _all_patterns;		
		assert(svm_check_parameter(&problem,&parameters)==NULL);
	}
	catch (...) {
		cleanup();
		throw fst_error("Classifier_LIBSVM allocation problem.");
	}
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(const PDataAccessor da)
{
	assert(da);
	try {
		cleanup();
		_all_patterns = da->getClassSizeSum(); assert(_all_patterns>0);
		_classes = da->getNoOfClasses(); assert(_classes>1);
		_features = da->getNoOfFeatures(); assert(_features>0);
		allocate();
	}
	catch (...) {
		cleanup();
		throw fst_error("Classifier_LIBSVM initialization problem.");
	}
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
void Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::cleanup()
{
	// NOTE: called from destructor
		if(problem.x!=NULL) {
			for(IDXTYPE p=0;p<_all_patterns;p++) if(problem.x[p]!=NULL) {free(problem.x[p]); problem.x[p]=NULL;}
			free(problem.x); problem.x=NULL; // the array only, not the actual pointers
		}
		if(problem.y!=NULL) free(problem.y); problem.y=NULL;
		if(onepattern!=NULL) free(onepattern); onepattern=NULL;
		if(parameters.weight_label!=NULL) free(parameters.weight_label); parameters.weight_label=NULL;
		if(parameters.weight!=NULL) free(parameters.weight); parameters.weight=NULL;
		if(model!=NULL) {svm_free_and_destroy_model(&model); model=NULL;}
		
		_all_patterns=0;
		_classes=0;
		_features=0;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::train(const PDataAccessor da, const PSubset sub)
{
	assert(da);
	assert(_classes==da->getNoOfClasses());
	assert(_features==da->getNoOfFeatures());
	assert(_all_patterns==da->getClassSizeSum());
	assert(problem.x!=NULL);
	assert(problem.y!=NULL);
	assert(onepattern!=NULL);
	assert(parameters.weight_label!=NULL);
	assert(parameters.weight!=NULL);
	assert(svm_check_parameter(&problem,&parameters)==NULL);
	try {
		// feature index buffering to accelerate data transfer
		DIMTYPE f,ftmp;
		bool b;
		if(!_index || _features<sub->get_n_raw()) _index.reset(new DIMTYPE[_features]);
		for(b=sub->getFirstFeature(f),_subfeatures=0;b==true;b=sub->getNextFeature(f),_subfeatures++) {assert(_subfeatures<_features); _index[_subfeatures]=f;}
#ifdef DEBUG
		{
			ostringstream sos;
			sos << "index: "; for(f=0;f<_subfeatures;f++) sos << _index[f] << " "; //sos << std::endl;
			syncout::print(std::cout,sos);
		}
#endif
		assert(_subfeatures==sub->get_d_raw());
		
		typename DATAACCESSOR::PPattern p,ptmp;
		IDXTYPE s,i;//,ifeatures;
		IDXTYPE count;
		DIMTYPE nzfeat;
		double tmp;
		const DIMTYPE da_train_loop=0; // to avoid mixup of get*Block() loops of different types
	
		count=0;
		for(DIMTYPE c_train=0;c_train<_classes;c_train++)
		{
			da->setClass(c_train);
			for(b=da->getFirstBlock(TRAIN,p,s,da_train_loop);b==true;b=da->getNextBlock(TRAIN,p,s,da_train_loop)) for(i=0;i<s;i++)
			{
				problem.y[count]=c_train; // current pattern class info
				nzfeat=0; // no. of nonzero features processed in current pattern
				ptmp=&p[i*_features]; // temp pointer to current pattern
				for(f=0;f<_subfeatures;f++)
				{
					ftmp=_index[f]; // (full set) feature index
					tmp=ptmp[ftmp]; // feature value
					if(tmp!=0.0) {
						problem.x[count][nzfeat].index=ftmp+1;
						problem.x[count][nzfeat].value=tmp; 
						nzfeat++;
					}
				}
				problem.x[count][nzfeat].index=-1;
				count++;
			}
		}
		assert(count>0);
		problem.l=count;
		if(model!=NULL) svm_free_and_destroy_model(&model);
		// train SVM model
		model=svm_train(&problem, &parameters);
	}
	catch(...) {
		cleanup();
		throw fst_error("Classifier_LIBSVM::train() error.");
	}	
	if(model==NULL) return false;
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::classify(DIMTYPE &cls, const PPattern &pattern)  // classifies pattern, returns the respective class index
{
	assert(model!=NULL);
	assert(svm_check_parameter(&problem,&parameters)==NULL);
	try {
		// assumes feature index buffering to accelerate data transfer
		assert(_index);
		assert(_subfeatures>0 && _subfeatures<=_features);
		
		DIMTYPE nzfeat;
		double tmp;
		RETURNTYPE tmpval;
		DIMTYPE ftmp;
	
		nzfeat=0; // no. of nonzero features processed in current pattern
		for(DIMTYPE f=0;f<_subfeatures;f++)
		{
			ftmp=_index[f]; // (full set) feature index
			tmp=pattern[ftmp]; // feature value
			if(tmp!=0.0) {
				onepattern[nzfeat].index=ftmp+1;
				onepattern[nzfeat].value=tmp; 
				nzfeat++;
			}
		}
		onepattern[nzfeat].index=-1;
		// now classify the pattern
		tmpval=svm_predict(model, onepattern);
		cls=(DIMTYPE)tmpval;
	}
	catch(...) {
		cleanup();
		throw fst_error("Classifier_LIBSVM::classify() error.");
	}	
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::test(RETURNTYPE &result, const PDataAccessor da)
{
	assert(da);
	assert(_classes==da->getNoOfClasses());
	assert(_features==da->getNoOfFeatures());
	assert(_all_patterns==da->getClassSizeSum());
	assert(model!=NULL);
	assert(svm_check_parameter(&problem,&parameters)==NULL);
	try {
		// assumes feature index buffering to accelerate data transfer
		assert(_index);
		assert(_subfeatures>0);
		
		typename DATAACCESSOR::PPattern p; //,ptmp;
		IDXTYPE s,i;
		IDXTYPE count, correct;
		DIMTYPE clstmp;
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
			ostringstream sos;
			sos << " result=" << result << std::endl;
			syncout::print(std::cout,sos);
		}
#endif
	}
	catch(...) {
		cleanup();
		throw fst_error("Classifier_LIBSVM::test() error.");
	}	
	return true;
}

} // namespace
#endif // FSTCLASSIFIERSVM_H ///:~
