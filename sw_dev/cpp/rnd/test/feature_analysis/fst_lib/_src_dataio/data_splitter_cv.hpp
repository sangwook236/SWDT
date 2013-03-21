#ifndef FSTDATASPLITTERCV_H
#define FSTDATASPLITTERCV_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    data_splitter_cv.hpp
   \brief   Implements train/test data splitting: by means of k-fold cross-validation
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at http://fst.utia.cz
   \date    October 2010
   \version 3.0.1.beta
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
#include <cstdlib> // rand()
#include <ctime> // time in srand()
#include <vector>
#include "error.hpp"
#include "global.hpp"
#include "data_splitter.hpp"

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

// CROSS-VALIDATION

//! Implements train/test data splitting: by means of k-fold cross-validation
template<class INTERVALCONTAINER, typename IDXTYPE>
class Data_Splitter_CV : public Data_Splitter<INTERVALCONTAINER,IDXTYPE> {
public:
	Data_Splitter_CV(const IDXTYPE _kfold) : Data_Splitter<INTERVALCONTAINER,IDXTYPE>(), kfold(_kfold) {assert(_kfold>1); notify("Data_Splitter_CV constructor.");}
	Data_Splitter_CV(const Data_Splitter_CV& dsp) : Data_Splitter<INTERVALCONTAINER,IDXTYPE>(dsp), kfold(dsp.kfold), tmpod(dsp.tmpod), tmpn(dsp.tmpn), tmpstartidx(dsp.tmpstartidx) {notify("Data_Splitter_CV copy constructor");}
	virtual ~Data_Splitter_CV() {notify("Data_Splitter_CV destructor.");}
	virtual IDXTYPE getNoOfSplits() const {return kfold;}
	virtual bool makeFirstSplit();
	virtual bool makeNextSplit();
	Data_Splitter_CV* stateless_clone() const {return new Data_Splitter_CV(*this);}
	virtual std::ostream& print(std::ostream& os) const {os << "Data_Splitter_CV(" << kfold << ")"; return os;}
private:
	typedef Data_Splitter<INTERVALCONTAINER,IDXTYPE> TCC;
	const IDXTYPE kfold;
	
	IDXTYPE tmpod;
	vector<IDXTYPE> tmpn;
	vector<IDXTYPE> tmpstartidx;
};

template<class INTERVALCONTAINER, typename IDXTYPE>
bool Data_Splitter_CV<INTERVALCONTAINER,IDXTYPE>::makeFirstSplit()
{
	TCC::assert_csplit();
	try {
		tmpn.resize(TCC::csplit.size());
		tmpstartidx.resize(TCC::csplit.size());
		unsigned int cidx=0;
		tmpod=kfold;
		for(typename TCC::ClassIterator c=TCC::csplit.begin();c!=TCC::csplit.end();c++) {
			tmpstartidx[cidx]=0;
			tmpn[cidx]=(*c)->n;

			const IDXTYPE tcs=tmpn[cidx]/tmpod;
			Data_Interval<IDXTYPE> tri={tcs,(*c)->n-tcs};	(*c)->train->clear();	(*c)->train->push_back(tri);
			Data_Interval<IDXTYPE> tei={0,tcs};	(*c)->test->clear();	(*c)->test->push_back(tei);
			
			tmpstartidx[cidx]+=tcs;
			tmpn[cidx]-=tcs;
			++cidx;
		}
		tmpod--;
	} catch(...) {
		throw fst_error("Data_Splitter_CV::makeFirstSplit() error.");
	}
	return true;
}

template<class INTERVALCONTAINER, typename IDXTYPE>
bool Data_Splitter_CV<INTERVALCONTAINER,IDXTYPE>::makeNextSplit()
{
	if(tmpod>0) {
	try {
		unsigned int cidx=0;
		for(typename TCC::ClassIterator c=TCC::csplit.begin();c!=TCC::csplit.end();c++) {
			const IDXTYPE tcs=tmpn[cidx]/tmpod;
			Data_Interval<IDXTYPE> tri={0,tmpstartidx[cidx]};	(*c)->train->clear();	(*c)->train->push_back(tri);
			if(tmpod>1) {Data_Interval<IDXTYPE> tri2={tmpstartidx[cidx]+tcs,(*c)->n-(tmpstartidx[cidx]+tcs)}; (*c)->train->push_back(tri2);}
			Data_Interval<IDXTYPE> tei={tmpstartidx[cidx],tcs};	(*c)->test->clear();	(*c)->test->push_back(tei);
			
			tmpstartidx[cidx]+=tcs;
			tmpn[cidx]-=tcs;
			++cidx;
		}
		tmpod--;
	} catch(...) {
		throw fst_error("Data_Splitter_CV::makeNextSplit() error.");
	}
	} else return false;
	return true;
}


} // namespace
#endif // FSTDATASPLITTERCV_H ///:~
