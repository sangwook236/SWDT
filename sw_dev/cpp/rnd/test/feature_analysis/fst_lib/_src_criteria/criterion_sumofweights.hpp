#ifndef FSTCRITERIONSUMOFWEIGHTS_H
#define FSTCRITERIONSUMOFWEIGHTS_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    criterion_sumofweights.hpp
   \brief   Criterion_Sum_Of_Weights returns sum of pre-specified feature weights for features in the evaluated subset
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
#include "error.hpp"
#include "global.hpp"
#include "criterion.hpp"

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

/*! \brief Returns sum of pre-specified feature weights for features in the evaluated subset.

    This trivial criterion is intended as secondary criterion to be used
    in conjunction with Result_Tracker_Regularizer, allowing to find subset
    among those close to the known best such that the sum of known feature weights 
    (e.g., feature acquisition cost) is minimized. This is usable, e.g., in medicine
    where different measurements have different costs (e.g., measuring body temperature 
    may be cheaper than laboratory tests, etc.). The technique is described in paper 
    "Somol, Grim, Pudil: The Problem of Fragile Feature Subset Preference in Feature Selection Methods 
    and A Proposal of Algorithmic Workaround. In Proc. ICPR 2010.  IEEE Computer 
    Society, 2010".
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET>
class Criterion_Sum_Of_Weights : public Criterion<RETURNTYPE,SUBSET> { // adapter class
public:
	typedef boost::shared_ptr<SUBSET> PSubset;
	typedef boost::shared_array<RETURNTYPE> PWeights;
	Criterion_Sum_Of_Weights() {notify("Criterion_Sum_Of_Weights constructor.");}
	virtual ~Criterion_Sum_Of_Weights() {notify("Criterion_Sum_Of_Weights destructor.");}

	virtual void initialize(const DIMTYPE noofweights, const PWeights weights) {_noofweights=noofweights; _weights=weights;}
	virtual void initialize(const DIMTYPE noofweights, const RETURNTYPE weights[]);
	
	virtual bool evaluate(RETURNTYPE &result, const PSubset sub); //!< sums up weights of the features selected in sub

	Criterion_Sum_Of_Weights* clone() const;
	Criterion_Sum_Of_Weights* sharing_clone() const {throw fst_error("Criterion_Sum_Of_Weights::sharing_clone() not supported, use Criterion_Sum_Of_Weights::clone() instead.");}
	Criterion_Sum_Of_Weights* stateless_clone() const {throw fst_error("Criterion_Sum_Of_Weights::stateless_clone() not supported, use Criterion_Sum_Of_Weights::clone() instead.");}
	
	virtual std::ostream& print(std::ostream& os) const {os << "Criterion_Sum_Of_Weights()"; return os;}
private:
	Criterion_Sum_Of_Weights(const Criterion_Sum_Of_Weights& css); // copy-constructor
protected:
	PWeights _weights;
	DIMTYPE _noofweights;
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET>
Criterion_Sum_Of_Weights<RETURNTYPE,DIMTYPE,SUBSET>::Criterion_Sum_Of_Weights(const Criterion_Sum_Of_Weights& css)
{
	notify("Criterion_Sum_Of_Weights copy-constructor.");
	if(css._weights)
	{
		assert(css._noofweights>0);
		_weights.reset(new RETURNTYPE[css._noofweights]);
		for(DIMTYPE i=0;i<css._noofweights;i++)_weights[i]=css._weights[i];
	}
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET>
void Criterion_Sum_Of_Weights<RETURNTYPE,DIMTYPE,SUBSET>::initialize(const DIMTYPE noofweights, const RETURNTYPE weights[])
{
	assert(noofweights>0);
	_noofweights=noofweights;
	_weights.reset(new RETURNTYPE[_noofweights]);
	for(DIMTYPE i=0;i<_noofweights;i++) _weights[i]=weights[i];
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET>
Criterion_Sum_Of_Weights<RETURNTYPE,DIMTYPE,SUBSET>* Criterion_Sum_Of_Weights<RETURNTYPE,DIMTYPE,SUBSET>::clone() const
{
	Criterion_Sum_Of_Weights<RETURNTYPE,DIMTYPE,SUBSET> *clone=new Criterion_Sum_Of_Weights<RETURNTYPE,DIMTYPE,SUBSET>(*this);
	clone->set_cloned();
	return clone;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET>
bool Criterion_Sum_Of_Weights<RETURNTYPE,DIMTYPE,SUBSET>::evaluate(RETURNTYPE &result, PSubset sub)
{
	assert(_weights);
	assert(sub);
	assert(_noofweights>=sub->get_n());
	assert(_noofweights==sub->get_n());

	result=0;
	DIMTYPE fidx;
	bool favail=sub->getFirstFeature(fidx);
	while(favail)
	{
		result+=_weights[fidx];
		favail=sub->getNextFeature(fidx);
	}
	return true;
}

} // namespace
#endif // FSTCRITERIONSUMOFWEIGHTS_H ///:~
