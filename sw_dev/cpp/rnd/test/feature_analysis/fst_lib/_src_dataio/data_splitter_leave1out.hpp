#ifndef FSTDATASPLITTERLEAVE1OUT_H
#define FSTDATASPLITTERLEAVE1OUT_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    data_splitter_leave1out.hpp
   \brief   Implements train/test data splitting: by means of leave-one-out
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

// LEAVE-ONE-OUT

//! Implements train/test data splitting: by means of leave-one-out
template<class INTERVALCONTAINER, typename IDXTYPE>
class Data_Splitter_Leave1Out : public Data_Splitter<INTERVALCONTAINER,IDXTYPE> {
public:
	Data_Splitter_Leave1Out() : Data_Splitter<INTERVALCONTAINER,IDXTYPE>() {notify("Data_Splitter_Leave1Out (empty) constructor.");}
	Data_Splitter_Leave1Out(const Data_Splitter_Leave1Out& dsp) : Data_Splitter<INTERVALCONTAINER,IDXTYPE>(dsp), pivot(dsp.pivot), pivotclass(dsp.pivotclass) {notify("Data_Splitter_Leave1Out copy constructor.");}
	virtual ~Data_Splitter_Leave1Out() {notify("Data_Splitter_Leave1Out destructor.");}
	virtual IDXTYPE getNoOfSplits() const;
	virtual bool makeFirstSplit();
	virtual bool makeNextSplit();
	Data_Splitter_Leave1Out* stateless_clone() const {return new Data_Splitter_Leave1Out(*this);}
	virtual std::ostream& print(std::ostream& os) const {os << "Data_Splitter_Leave1Out()"; return os;}
private:
	typedef Data_Splitter<INTERVALCONTAINER,IDXTYPE> TCC;
	IDXTYPE pivot; // which sample to exclude..
	typename TCC::ClassIterator pivotclass; // ..in which class
};

template<class INTERVALCONTAINER, typename IDXTYPE>
IDXTYPE Data_Splitter_Leave1Out<INTERVALCONTAINER,IDXTYPE>::getNoOfSplits() const
{
	IDXTYPE sum=0;
	for(typename TCC::ClassIterator c=TCC::csplit.begin();c!=TCC::csplit.end();c++) sum+=(*c)->n;
	return sum;
}

template<class INTERVALCONTAINER, typename IDXTYPE>
bool Data_Splitter_Leave1Out<INTERVALCONTAINER,IDXTYPE>::makeFirstSplit()
{
	TCC::assert_csplit();
	try {
		pivotclass=TCC::csplit.begin();
		while(pivotclass!=TCC::csplit.end() && (*pivotclass)->n==0) pivotclass++;
		if(pivotclass!=TCC::csplit.end()) {
			pivot=0;
			for(typename TCC::ClassIterator c=TCC::csplit.begin();c!=TCC::csplit.end();c++) {
				if(c==pivotclass) {
					Data_Interval<IDXTYPE> tri={pivot+1,(*c)->n-1};	(*c)->train->clear();	(*c)->train->push_back(tri);
					Data_Interval<IDXTYPE> tei={pivot,1};	(*c)->test->clear();	(*c)->test->push_back(tei);
				} else {
					Data_Interval<IDXTYPE> tri={0,(*c)->n};	(*c)->train->clear();	(*c)->train->push_back(tri); // all
					(*c)->test->clear();
				}
			}
			++pivot;
		} else return false;
	} catch(...) {
		throw fst_error("Data_Splitter_Leave1Out::makeFirstSplit() error.");
	}
	return true;
}

template<class INTERVALCONTAINER, typename IDXTYPE>
bool Data_Splitter_Leave1Out<INTERVALCONTAINER,IDXTYPE>::makeNextSplit()
{
	if(pivotclass!=TCC::csplit.end()) {
	try {
		if(pivot>=(*pivotclass)->n) {
			pivot=0;
			pivotclass++;
			while(pivotclass!=TCC::csplit.end() && (*pivotclass)->n==0) pivotclass++;
			if(pivotclass==TCC::csplit.end()) return false;
		}
		for(typename TCC::ClassIterator c=TCC::csplit.begin();c!=TCC::csplit.end();c++) {
			if(c==pivotclass) {
				(*c)->train->clear();
				if(pivot>0) {Data_Interval<IDXTYPE> tri={0,pivot};	(*c)->train->push_back(tri);}
				if(pivot<(*pivotclass)->n-1) {Data_Interval<IDXTYPE> tri2={pivot+1,(*c)->n-(pivot+1)};	(*c)->train->push_back(tri2);}
				Data_Interval<IDXTYPE> tei={pivot,1};	(*c)->test->clear();	(*c)->test->push_back(tei);
			} else {
				Data_Interval<IDXTYPE> tri={0,(*c)->n};	(*c)->train->clear();	(*c)->train->push_back(tri); // all
				(*c)->test->clear();
			}
		}
		++pivot;
	} catch(...) {
		throw fst_error("Data_Splitter_Leave1Out::makeNextSplit() error.");
	}
	} else return false;
	return true;
}


} // namespace
#endif // FSTDATASPLITTERLEAVE1OUT_H ///:~
