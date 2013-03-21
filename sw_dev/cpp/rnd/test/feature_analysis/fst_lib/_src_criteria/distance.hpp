#ifndef FSTDISTANCE_H
#define FSTDISTANCE_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    distance.hpp
   \brief   Defines interface for distance evaluators (to be used primarily in Classifier_kNN)
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
#include <cmath>
#include <list>
#include <cstring> // memcpy
#include "error.hpp"
#include "global.hpp"
#include "clonable.hpp"

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

//! Abstract class, defines interface for distance evaluators to be used in Classifier_kNN
template<typename DATATYPE, typename DIMTYPE, class SUBSET>
class Distance : public Clonable {
public:
	typedef boost::shared_ptr<SUBSET> PSubset;
	Distance() : _n(0), _d(0) {}
	Distance(const Distance &d);
	virtual ~Distance() {}
	void reset(const DIMTYPE n);
	
	void narrow_to(const PSubset sub); // fill _index[] -> access to data is then mapped to sub-matrix
	void denarrow(); // reset _index[] -> access is reset to the full matrix

	DIMTYPE get_n() const {return _n;}
	DIMTYPE get_d() const {return _d;}

	virtual DATATYPE distance(const DATATYPE *pattern1, const DATATYPE *pattern2) =0;

	Distance* sharing_clone() const {throw fst_error("Distance::sharing_clone() not supported, use Distance::clone() instead.");}
	Distance* stateless_clone() const {throw fst_error("Distance::stateless_clone() not supported, use Distance::clone() instead.");}
protected:
	DIMTYPE _n;
	DIMTYPE _d;
	boost::scoped_array<DIMTYPE> _index;
};

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
Distance<DATATYPE,DIMTYPE,SUBSET>::Distance(const Distance& d) :
	_n(d._n),
	_d(d._d)
{
	notify("Distance copy-constructor.");
	if(_n>0) {_index.reset(new DIMTYPE[_n]); memcpy((void *)_index.get(),(void *)(d._index).get(),sizeof(DIMTYPE)*_n);}
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Distance<DATATYPE,DIMTYPE,SUBSET>::reset(const DIMTYPE n)
{
	assert(n>0);
	_n=n;
	_index.reset(new DIMTYPE[_n]);
	for(DIMTYPE i=0;i<_n;i++) _index[i]=i;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Distance<DATATYPE,DIMTYPE,SUBSET>::narrow_to(const PSubset sub)
{
	assert(_n>=sub->get_d_raw()); //enough space to store subset ?
	assert(_index);
	DIMTYPE f;
	_d=0;
	for(bool b=sub->getFirstFeature(f);b!=false;b=sub->getNextFeature(f)) _index[_d++]=f;
	assert(_d>0);
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Distance<DATATYPE,DIMTYPE,SUBSET>::denarrow()
{
	assert(_index);
	for(DIMTYPE i=0;i<_n;i++) _index[i]=i;
	_d=_n;
}


} // namespace
#endif // FSTDISTANCE_H ///:~
