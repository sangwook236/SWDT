#ifndef FSTCRITERIONWRAPPERBIASESTIMATE_H
#define FSTCRITERIONWRAPPERBIASESTIMATE_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    criterion_wrapper_bias_estimate.hpp
   \brief   Wrapper Bias Evaluator - returns classifier bias estimate (difference between accuracy on training and validation data parts)
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
#include <cmath>
#include "error.hpp"
#include "global.hpp"
#include "criterion_wrapper.hpp"

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

/*! \brief Wrapper Bias Evaluator - returns bias estimate (difference of classifier accuracy on training and validation data parts)

   \detailed Wraps Classifier objects to serve as secondary feature selection criterion.
   Trains classifier on training data, then evaluates the difference between estimated classification 
   accuracy on training data and test data. Especially in small-sample scenarios a considerable difference
   (i.e., bias) can be expected. Large bias can be viewed as a sign of overfitting and can thus
   be used as indicator of feature selection process failure.
      
   \note Expect at least 2x slower operation than that of Criterion_Wrapper.
*/
template<class RETURNTYPE, class SUBSET, class CLASSIFIER, class DATAACCESSOR>
class Criterion_Wrapper_Bias_Estimate : public Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR> { // adapter class
public:
	typedef Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR> parent;
	typedef boost::shared_ptr<CLASSIFIER> PClassifier;
	typedef boost::shared_ptr<DATAACCESSOR> PDataAccessor;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Criterion_Wrapper_Bias_Estimate():_collected_bias_estimate(0.0),_collected_bias_count(0.0) {notify("Criterion_Wrapper_Bias_Estimate constructor.");}
	virtual ~Criterion_Wrapper_Bias_Estimate() {notify("Criterion_Wrapper_Bias_Estimate destructor.");}

	//\note parent initialize() must be called before the first evaluate() call

	virtual bool evaluate(RETURNTYPE &result, const PSubset sub); //!< trains then tests then returns estimated classifcation accuracy

	RETURNTYPE getBiasEstimate() const {if(_collected_bias_count>0.0) return _collected_bias_estimate/_collected_bias_count; else return 0.0;} //!< returns estimated bias average over all evaluate() invocations since the last reset
	void resetBiasEstimate() {_collected_bias_estimate=0.0; _collected_bias_count=0.0;}

	Criterion_Wrapper_Bias_Estimate* clone() const;
	Criterion_Wrapper_Bias_Estimate* sharing_clone() const {throw fst_error("Criterion_Wrapper_Bias_Estimate::sharing_clone() not supported, use Criterion_Wrapper_Bias_Estimate::clone() instead.");}
	Criterion_Wrapper_Bias_Estimate* stateless_clone() const {throw fst_error("Criterion_Wrapper_Bias_Estimate::stateless_clone() not supported, use Criterion_Wrapper_Bias_Estimate::clone() instead.");}
	
	virtual std::ostream& print(std::ostream& os) const {os << "Criterion_Wrapper_Bias_Estimate()"; if(parent::_classifier) os << " with " << *parent::_classifier; return os;}
protected:
	Criterion_Wrapper_Bias_Estimate(const Criterion_Wrapper_Bias_Estimate& cwbe) : Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR>(cwbe), _collected_bias_estimate(cwbe._collected_bias_estimate), _collected_bias_count(cwbe._collected_bias_count) {} // copy-constructor (clones _classifier and _da)

	RETURNTYPE _collected_bias_estimate;
	RETURNTYPE _collected_bias_count;
};

template<class RETURNTYPE, class SUBSET, class CLASSIFIER, class DATAACCESSOR>
Criterion_Wrapper_Bias_Estimate<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR>* Criterion_Wrapper_Bias_Estimate<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR>::clone() const
{
	Criterion_Wrapper_Bias_Estimate<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR> *clone=new Criterion_Wrapper_Bias_Estimate<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR>(*this);
	clone->set_cloned();
	return clone;
}

template<class RETURNTYPE, class SUBSET, class CLASSIFIER, class DATAACCESSOR>
bool Criterion_Wrapper_Bias_Estimate<RETURNTYPE,SUBSET,CLASSIFIER,DATAACCESSOR>::evaluate(RETURNTYPE &result, PSubset sub)
{
	assert(parent::_da);
	assert(parent::_classifier);
	RETURNTYPE val1, val2, bias=0.0;
	result=0.0;
	RETURNTYPE cnt=0.0;
	// bias estimation
	parent::_da->resubstitute();
	for(bool b=parent::_da->getFirstSplit();b==true;b=parent::_da->getNextSplit()) {
		// train on TRAIN data part
		if(!parent::_classifier->train(parent::_da,sub)) return false;
		// test on TRAIN (dependent) data part
		parent::_da->substitute(TEST,TRAIN);
		if(!parent::_classifier->test(val2,parent::_da)) return false;
		// test on TEST (independent) data part
		parent::_da->substitute(TEST,TEST);
		if(!parent::_classifier->test(val1,parent::_da)) return false;

		bias+=val2-val1;
		cnt+=1.0;
	}
	_collected_bias_estimate+=bias;
	_collected_bias_count+=(RETURNTYPE)cnt;
	result=bias/(RETURNTYPE)cnt;
	return true;
}

} // namespace
#endif // FSTCRITERIONWRAPPERBIASESTIMATE_H ///:~
