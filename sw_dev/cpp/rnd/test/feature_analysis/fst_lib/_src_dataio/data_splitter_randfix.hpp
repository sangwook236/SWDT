#ifndef FSTDATASPLITTERRANDFIX_H
#define FSTDATASPLITTERRANDFIX_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    data_splitter_randfix.hpp
   \brief   Implements train/test data splitting: equal to Data_Splitter_RandomRandom except that the same test data is returned in each loop
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
#include "data_splitter_randrand.hpp"

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

// NOTE: this is a special-purpose modification of Data_Splitter_RandomRandom
//       that generates train data equally to Data_Splitter_RandomRandom
//       but selects always the same test data (consecutive interval at the end of each class)

//! Implements train/test data splitting: equal to Data_Splitter_RandomRandom except that the same test data is returned in each loop
template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
class Data_Splitter_TrainRandom_TestFixed : public Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE> {
public:
	typedef Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE> parent;
	Data_Splitter_TrainRandom_TestFixed(const IDXTYPE _splits, const IDXTYPE _perctrain, const IDXTYPE _perctest, const bool randomize=false) : Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE>(_splits, _perctrain, _perctest, randomize) {notify("Data_Splitter_TrainRandom_TestFixed constructor.");}
	Data_Splitter_TrainRandom_TestFixed(const Data_Splitter_TrainRandom_TestFixed& dsp) : Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE>(dsp) {notify("Data_Splitter_TrainRandom_TestFixed copy constructor.");}
	virtual ~Data_Splitter_TrainRandom_TestFixed() {notify("Data_Splitter_TrainRandom_TestFixed destructor.");}
	Data_Splitter_TrainRandom_TestFixed* stateless_clone() const {return new Data_Splitter_TrainRandom_TestFixed(*this);}
	virtual std::ostream& print(std::ostream& os) const {os << "Data_Splitter_TrainRandom_TestFixed(splits=" << parent::splits << ", %train=" << parent::perctrain << ", %test=" << parent::perctest << ")"; return os;}
protected:
	typedef Data_Splitter<INTERVALCONTAINER,IDXTYPE> TCC;
	virtual void makeRandomSplit(const IDXTYPE n, const boost::shared_ptr<INTERVALCONTAINER> list_train, const boost::shared_ptr<INTERVALCONTAINER> list_test);
};

template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
void Data_Splitter_TrainRandom_TestFixed<INTERVALCONTAINER,IDXTYPE,BINTYPE>::makeRandomSplit(const IDXTYPE n, const boost::shared_ptr<INTERVALCONTAINER> list_train, const boost::shared_ptr<INTERVALCONTAINER> list_test)
{
	try {
		assert(list_train);
		assert(list_test);
		assert(n>0);
		assert(0<=parent::perctrain);
		assert(0<=parent::perctest);
		assert(parent::perctrain+parent::perctest<=100);
		if(!parent::_data || parent::_n_max<n) {parent::_data.reset(new BINTYPE[n]); parent::_n_max=n;}
		const BINTYPE id_empty = 0;
		const BINTYPE id_train = 1;
		const IDXTYPE tesiz=(n * parent::perctest)/100;
		const IDXTYPE trsiz= (parent::perctrain+parent::perctest==100) ? n-tesiz : (n * parent::perctrain)/100;
		assert(trsiz+tesiz<=n);
		IDXTYPE i;
		for(i=0;i<n;i++) parent::_data[i]=id_empty;

		// now fill randomly perctrain% for training and perctest% for testing
		// NOTE: acceleration - rand() needs to be called only min{perctrain,100-perctrain} and min{perctest,100-perctest} times - just adjust _data pre-fill
		//                    - also, if(perctrain+perctest==100), then rand() loop is needed only for one of them
		if(parent::perctrain<=(100-parent::perctest)/2) {
			parent::fill_randomly(n,id_empty,id_train,trsiz/*count*/,0/*minidx*/,n-1-tesiz/*maxidx*/); // randomly choose train samples
		} else {
			parent::fill(n,id_empty,id_train,0/*minidx*/,n-1-tesiz/*maxidx*/);
			parent::fill_randomly(n,id_train,id_empty,n-tesiz-trsiz/*count*/,0/*minidx*/,n-1-tesiz/*maxidx*/); // randomly choose non-train samples
		}

		// now transform the binary representation to TCC::train list and create TCC::test list
		parent::translate(n,id_train,list_train);
		list_test->clear();
		if(tesiz>0) {Data_Interval<IDXTYPE> tin={n-tesiz,tesiz}; list_test->push_back(tin);}

	} catch(...) {
		throw fst_error("Data_Splitter_TrainRandom_TestFixed::makeRandomSplit() error.");
	}
}


} // namespace
#endif // FSTDATASPLITTERRANDFIX_H ///:~
