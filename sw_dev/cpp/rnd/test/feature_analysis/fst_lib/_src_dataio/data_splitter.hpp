#ifndef FSTDATASPLITTER_H
#define FSTDATASPLITTER_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    data_splitter.hpp
   \brief   Defines interface for data splitting implementations (for use in data accessors)
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

template<class INTERVALCONTAINER, typename IDXTYPE>
class Data_Splitter;

template<class INTERVALCONTAINER, typename IDXTYPE>
std::ostream& operator<<(std::ostream& os, const Data_Splitter<INTERVALCONTAINER,IDXTYPE>& ds);

//! Abstract class, defines interface for data splitting implementations (for use in data accessors)
template<class INTERVALCONTAINER, typename IDXTYPE>
class Data_Splitter : public Clonable {
	friend std::ostream& operator<< <>(std::ostream& os, const Data_Splitter<INTERVALCONTAINER,IDXTYPE>& ds);
public:
	typedef boost::shared_ptr<INTERVALCONTAINER> PIntervaller;
	Data_Splitter() {csplit.clear(); notify("Data_Splitter constructor.");}
	Data_Splitter(const Data_Splitter& dsp);
	virtual ~Data_Splitter() {notify("Data_Splitter destructor.");}

	void assign(const unsigned int _clsidx, const PIntervaller _train, const PIntervaller _test);
	void set_n(const unsigned int _clsidx, const IDXTYPE _n);
	IDXTYPE get_n(const unsigned int _clsidx) const;

	virtual IDXTYPE getNoOfSplits() const =0;	// may not be needed /??implemented
	virtual bool makeFirstSplit()=0;
	virtual bool makeNextSplit()=0;
	
	Data_Splitter* clone() const {throw fst_error("Data_Splitter::clone() not supported, use Data_Splitter::stateless_clone() instead.");}
	Data_Splitter* sharing_clone() const {throw fst_error("Data_Splitter::sharing_clone() not supported, use Data_Splitter::stateless_clone() instead.");}

	virtual std::ostream& print(std::ostream& os) const {return os;};

protected:
	//! Data splitting support structure; holds one set of intervals (train, test) per each (data)class
	class TClassSplitter {
	public:
		TClassSplitter() {n=0; train.reset(); test.reset(); notify("TClassSplitter empty constructor.");}
		TClassSplitter(const TClassSplitter& cs) : n(cs.n) {
			notify("TClassSplitter strong copy constructor begin.");
			if(cs.train) train.reset(new INTERVALCONTAINER(*cs.train));
			if(cs.test) test.reset(new INTERVALCONTAINER(*cs.test));
			notify("TClassSplitter strong copy constructor end.");
		}
		~TClassSplitter() {notify("TClassSplitter destructor.");}
		IDXTYPE n;	//!< n=total no. of patterns
		PIntervaller train;	//!< pointer to external list of intervals that will be overwritten from here
		PIntervaller test;	//!< pointer to external list of intervals that will be overwritten from here
	};
	typedef boost::shared_ptr<TClassSplitter> PClassSplitter;
	std::vector<PClassSplitter> csplit; // one set of intervals per each class
	typedef typename std::vector<PClassSplitter>::const_iterator ClassIterator;

	void assert_csplit() const; // for debugging purposes
};


template<class INTERVALCONTAINER, typename IDXTYPE>
Data_Splitter<INTERVALCONTAINER,IDXTYPE>::Data_Splitter(const Data_Splitter& dsp)
{
	notify("Data_Splitter strong copy constructor begin.");
	csplit.clear();
	for(typename std::vector<PClassSplitter>::const_iterator c=dsp.csplit.begin();c!=dsp.csplit.end();c++) 
		{PClassSplitter pcs(new TClassSplitter(*(*c))); this->csplit.push_back(pcs);}
	notify("Data_Splitter strong copy constructor end.");
}

template<class INTERVALCONTAINER, typename IDXTYPE>
void Data_Splitter<INTERVALCONTAINER,IDXTYPE>::assign(const unsigned int _clsidx, const PIntervaller _train, const PIntervaller _test)
{
	assert(_clsidx>=0);
	assert(_train); 
	assert(_test);
	try {	
		while(csplit.size()<=_clsidx) {PClassSplitter pcs(new TClassSplitter); csplit.push_back(pcs);}
		csplit[_clsidx]->train=_train; 
		csplit[_clsidx]->test=_test;
	} catch(...) {
		throw fst_error("Data_Splitter::assign() out-of-index? error.");
	}
}
	
template<class INTERVALCONTAINER, typename IDXTYPE>
void Data_Splitter<INTERVALCONTAINER,IDXTYPE>::set_n(const unsigned int _clsidx, const IDXTYPE _n)
{
	assert(_n>0); 
	assert(_clsidx>=0);
	try {	
		while(csplit.size()<=_clsidx) {PClassSplitter pcs(new TClassSplitter); csplit.push_back(pcs);}
		csplit[_clsidx]->n=_n;
	} catch(...) {
		throw fst_error("Data_Splitter::assign() out-of-index? error.");
	}
}

template<class INTERVALCONTAINER, typename IDXTYPE>
IDXTYPE Data_Splitter<INTERVALCONTAINER,IDXTYPE>::get_n(const unsigned int _clsidx) const
{
	assert(_clsidx<csplit.size());
	try {
		return csplit[_clsidx]->n;
	} catch(...) {
		throw fst_error("Data_Splitter::get_n() out-of-index? error.");
	}
}

template<class INTERVALCONTAINER, typename IDXTYPE>
std::ostream& operator<<(std::ostream& os, const Data_Splitter<INTERVALCONTAINER,IDXTYPE>& ds) {
	return ds.print(os);
}

template<class INTERVALCONTAINER, typename IDXTYPE>
void Data_Splitter<INTERVALCONTAINER,IDXTYPE>::assert_csplit() const
{
#ifdef DEBUG
	assert(csplit.size()>0);
	for(ClassIterator c=csplit.begin();c!=csplit.end();c++) {
		assert((*c)->n>0);
		assert((*c)->train);
		assert((*c)->test);
	}
#endif
}


} // namespace
#endif // FSTDATASPLITTER_H ///:~
