#ifndef FSTCRITERIONMULTINOMBHATTACHARYYA_H
#define FSTCRITERIONMULTINOMBHATTACHARYYA_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    criterion_multinom_bhattacharyya.hpp
   \brief   Implements Bhattacharyya distance based on multinomial model to serve as feature selection criterion
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
#include "error.hpp"
#include "global.hpp"
#include "criterion_multinom.hpp"
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

//! Implements Bhattacharyya distance based on multinomial model to serve as feature selection criterion
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Criterion_Multinomial_Bhattacharyya : public Criterion_Multinomial<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> {
public:
	typedef Criterion_Multinomial<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent;
	typedef boost::shared_ptr<DATAACCESSOR> PDataAccessor;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Criterion_Multinomial_Bhattacharyya() {_IB_computed=false; notify("Criterion_Multinomial_Bhattacharyya constructor.");}
	virtual ~Criterion_Multinomial_Bhattacharyya() {notify("Criterion_Multinomial_Bhattacharyya destructor.");}

	virtual bool evaluate(RETURNTYPE &result, const PSubset sub);
	virtual bool initialize(PDataAccessor da); 

	Criterion_Multinomial_Bhattacharyya* clone() const;
	Criterion_Multinomial_Bhattacharyya* sharing_clone() const {throw fst_error("Criterion_Multinomial_Bhattacharyya::sharing_clone() not supported, use Criterion_Multinomial_Bhattacharyya::clone() instead.");}
	Criterion_Multinomial_Bhattacharyya* stateless_clone() const {throw fst_error("Criterion_Multinomial_Bhattacharyya::stateless_clone() not supported, use Criterion_Multinomial_Bhattacharyya::clone() instead.");}
	
	virtual std::ostream& print(std::ostream& os) const {os << "Criterion_Multinomial_Bhattacharyya()"; return os;}
private:
	Criterion_Multinomial_Bhattacharyya(const Criterion_Multinomial_Bhattacharyya& cmb); // copy-constructor
private:
	bool _IB_computed;
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Criterion_Multinomial_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Criterion_Multinomial_Bhattacharyya(const Criterion_Multinomial_Bhattacharyya& cmb) :
	Criterion_Multinomial<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(cmb),
	_IB_computed(cmb._IB_computed)
{
	notify("Criterion_Multinomial_Bhattacharyya copy-constructor.");
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Criterion_Multinomial_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>* Criterion_Multinomial_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::clone() const
{
	Criterion_Multinomial_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> *clone=new Criterion_Multinomial_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(*this);
	clone->set_cloned();
	return clone;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Multinomial_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::evaluate(RETURNTYPE &result, PSubset sub)
{
	notify("Criterion_Multinomial_Bhattacharyya::evaluate().");
	assert(parent::_model);
	assert(parent::get_n()>0);
	assert(sub);
	if(sub->get_d_raw()==0) return false;
	DIMTYPE _d = sub->get_d_raw();
	DIMTYPE f;

	if(_d==1) {// general form of Bhattacharyya unusable, return Indivdual Bhattacharyya
		if(!_IB_computed) {
			parent::_model->denarrow();
			parent::_model->compute_theta();
			parent::_model->compute_IB();
			_IB_computed=true;	
		}
		if(!sub->getFirstFeature(f)) return false; assert(f>=0 && f<parent::_model->get_n());
		result = parent::_model->get_IB()[f]; // individual Bhatt - higher values denote better terms
		bool b=sub->getNextFeature(f); // just to finish the loop inside sub (to prevent future asserts in sub->set_forward_mode etc.)
		assert(b==false);
	} else {
		parent::_model->narrow_to(sub);
		parent::_model->compute_theta();
		REALTYPE doc_avg_length = parent::_model->get_doc_avg_length(); // valid after compute_theta() call
		DIMTYPE _classes=parent::_model->get_classes();
		
		REALTYPE *tmpoint1, *tmpoint2;
		RETURNTYPE thetasum;
		RETURNTYPE value=0.0;
		DIMTYPE c1,c2;
#ifdef DEBUG
		{
			ostringstream sos; sos << "d="<<_d<<", doc_avg_length="<<doc_avg_length<<std::endl;
			syncout::print(std::cout,sos);
		}
#endif
		for(c1=0;c1<_classes;c1++) for(c2=c1+1;c2<_classes;c2++) // for class pair c1,c2
		{
			tmpoint1=&(parent::_model->get_theta()[c1*_d]);
			tmpoint2=&(parent::_model->get_theta()[c2*_d]);
			thetasum=0.0;
			for(f=0;f<_d;f++) 
			{
				thetasum+=sqrt(tmpoint1[f]*tmpoint2[f]);
#ifdef DEBUG
				{
					ostringstream sos; sos << "classes<"<<c1<<","<<c2<<">: feat "<<f<<".thetasum="<<thetasum<<", tmpoint1[i="<<f<<"]="<< tmpoint1[f]<< ", tmpoint2[i="<<f<<"]="<< tmpoint2[f]<<std::endl;
					syncout::print(std::cout,sos);
				}
#endif
			}
			value+= (double)log(thetasum) * parent::_model->get_Pc(c1) * parent::_model->get_Pc(c2); // weighted
		}
		result = (-doc_avg_length)*value;
#ifdef DEBUG
		{
			ostringstream sos; sos << "result="<<result<<std::endl<<std::endl;
			syncout::print(std::cout,sos);
		}
#endif
	}
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Multinomial_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(PDataAccessor da)
{
	notify("Criterion_Multinomial_Bhattacharyya::initialize().");
	parent::initialize(da);
	_IB_computed=false;
	return true; 
}


//----------------------------------------------------------------------------

/*! \brief Implements individual Mutual Information based on multinomial model to serve as feature selection criterion in Best Individual Feature setting (feature ranking) only
    \note Can be used to evaluate single features only ! */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Criterion_Multinomial_MI : public Criterion_Multinomial<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> {
public:
	typedef Criterion_Multinomial<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent;
	typedef boost::shared_ptr<DATAACCESSOR> PDataAccessor;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Criterion_Multinomial_MI() {_MI_computed=false; notify("Criterion_Multinomial_MI constructor.");}
	virtual ~Criterion_Multinomial_MI() {notify("Criterion_Multinomial_MI destructor.");}

	virtual bool evaluate(RETURNTYPE &result, const PSubset sub);
	virtual bool initialize(PDataAccessor da); 
private:
	bool _MI_computed;
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Multinomial_MI<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::evaluate(RETURNTYPE &result, PSubset sub)
{
	notify("Criterion_Multinomial_MI::evaluate().");
	assert(parent::_model);
	assert(parent::get_n()>0);
	assert(sub);
	if(sub->get_d_raw()==0) return false;
	DIMTYPE _d = sub->get_d_raw();
	DIMTYPE f;

	assert(_d==1);
	
	if(!_MI_computed) {
		parent::_model->denarrow();
		parent::_model->compute_theta();
		parent::_model->compute_MI();
		_MI_computed=true;	
	}
	if(!sub->getFirstFeature(f)) return false; assert(f>=0 && f<parent::_model->get_n());
	result = parent::_model->get_MI()[f]; // individual Bhatt - higher values denote better terms
	bool b=sub->getNextFeature(f); // just to finish the loop inside sub (to prevent future asserts in sub->set_forward_mode etc.)
	assert(b==false);
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Multinomial_MI<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(PDataAccessor da)
{
	notify("Criterion_Multinomial_MI::initialize().");
	parent::initialize(da);
	_MI_computed=false;
	return true; 
}

} // namespace
#endif // FSTCRITERIONMULTINOMBHATTACHARYYA_H ///:~
