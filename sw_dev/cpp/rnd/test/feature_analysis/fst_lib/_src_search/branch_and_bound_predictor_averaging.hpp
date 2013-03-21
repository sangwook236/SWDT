#ifndef FSTSEARCHBRANCHANDBOUNDPREDICTORAVERAGING_H
#define FSTSEARCHBRANCHANDBOUNDPREDICTORAVERAGING_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    branch_and_bound_predictor_averaging.hpp
   \brief   Averaging Prediction Mechanism for use in Fast Branch & Bound and Branch & Bound with Partial Prediction
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
  * FST3 software is _available free of charge for non-commercial use. 
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
#include "error.hpp"
#include "global.hpp"
#include "branch_and_bound_predictor.hpp"

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

/*! \brief Averaging Prediction Mechanism in Fast Branch and Bound & Branch and Bound with Partial Prediction
*/
template<class RETURNTYPE, typename DIMTYPE>
class Branch_And_Bound_Predictor_Averaging : public Branch_And_Bound_Predictor<RETURNTYPE,DIMTYPE> {
public:
	Branch_And_Bound_Predictor_Averaging(): _gamma(1.0), _delta(1) {notify("Branch_And_Bound_Predictor_Averaging constructor.");}
	virtual ~Branch_And_Bound_Predictor_Averaging() {notify("Branch_And_Bound_Predictor_Averaging destructor.");}

	virtual void initialize(const DIMTYPE features);
	virtual bool learn(const DIMTYPE feature, const RETURNTYPE value);
	virtual bool predict(const DIMTYPE feature, RETURNTYPE &value) const;
			unsigned long get_count(const DIMTYPE feature) const {assert(0<=feature && feature<_count.size()); return _count[feature];}
	
			void set_gamma(const RETURNTYPE gamma=1.0) {assert(gamma>=0.0); _gamma=gamma;}
			void set_delta(const unsigned int delta=1) {assert(delta>=1); _delta=delta;}
	
	virtual std::ostream& print(std::ostream& os) const;
protected:
	vector<RETURNTYPE>    _value; //!< sums up 'learned' values
	vector<unsigned long> _count; //!< records the number of summed up values
	
	RETURNTYPE    _gamma;
	unsigned int  _delta;
};

template<class RETURNTYPE, typename DIMTYPE>
void Branch_And_Bound_Predictor_Averaging<RETURNTYPE,DIMTYPE>::initialize(const DIMTYPE features)
{
	assert(features>=1);
	_value.clear(); _value.resize(features);
	_count.clear(); _count.resize(features);
	for(DIMTYPE i=0;i<features;i++) {_value[i]=0.0; _count[i]=0;}
}

template<class RETURNTYPE, typename DIMTYPE>
bool Branch_And_Bound_Predictor_Averaging<RETURNTYPE,DIMTYPE>::learn(const DIMTYPE feature, const RETURNTYPE value)
{
	assert(value>=0.0);
	assert(feature>=0 && feature<_value.size());
	_value[feature]+=value;
	_count[feature]++;
	return true;
}

template<class RETURNTYPE, typename DIMTYPE>
bool Branch_And_Bound_Predictor_Averaging<RETURNTYPE,DIMTYPE>::predict(const DIMTYPE feature, RETURNTYPE &value) const
{
	assert(feature>=0 && feature<_value.size());
	if(_count[feature]>=_delta) {	
		value=_gamma*(_value[feature]/(RETURNTYPE)_count[feature]);
		return true;
	}
	return false;
}

template<class RETURNTYPE, typename DIMTYPE>
std::ostream& Branch_And_Bound_Predictor_Averaging<RETURNTYPE,DIMTYPE>::print(std::ostream& os) const
{
	os << "Branch_And_Bound_Predictor_Averaging(gamma="<<_gamma<<", delta="<<_delta<<")"; 
	RETURNTYPE val;
	unsigned long cnt=0;
	for(DIMTYPE i=0;i<_value.size();i++) if(_count[i]>0) {
		if(predict(i,val)) os << std::endl << "feature "<<i<<": predictor value="<<val<< ", count="<<_count[i];
		cnt+=_count[i];
	}
	if(cnt>0) os << std::endl << "Total count: "<<cnt;
	return os;
}

} // namespace
#endif // FSTSEARCHBRANCHANDBOUNDPREDICTORAVERAGING_H ///:~
