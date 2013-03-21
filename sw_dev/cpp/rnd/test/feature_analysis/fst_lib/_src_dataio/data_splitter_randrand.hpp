#ifndef FSTDATASPLITTERRANDRAND_H
#define FSTDATASPLITTERRANDRAND_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    data_splitter_randrand.hpp
   \brief   Implements train/test data splitting: use randomly chosen x% of data samples for training and another y% of data for testing, without overlaps
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

// NOTE: generates non-overlapping TCC::train and TCC::test lists of randomly chosen samples
// NOTE: repeated traversal through split loops does not give equal output ! 
//       (i.e., generated random sample subsets are not remembered, thus two consecutive makeFirstSplit() calls give different output)

//! Implements train/test data splitting: use randomly chosen x% of data samples for training and another y% of data for testing, without overlaps, separately in each class
template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
class Data_Splitter_RandomRandom : public Data_Splitter<INTERVALCONTAINER,IDXTYPE> {
public:
	Data_Splitter_RandomRandom(const IDXTYPE _splits, const IDXTYPE _perctrain, const IDXTYPE _perctest, const bool randomize=false) : Data_Splitter<INTERVALCONTAINER,IDXTYPE>() ,_n_max(0), perctrain(_perctrain), perctest(_perctest), splits(_splits), _randomize(randomize) {assert(splits>0); assert(0<=perctrain); assert(0<=perctest); assert(perctrain+perctest<=100); current_split=0; notify("Data_Splitter_RandomRandom constructor.");}
	Data_Splitter_RandomRandom(const Data_Splitter_RandomRandom& dsp) : Data_Splitter<INTERVALCONTAINER,IDXTYPE>(dsp) ,_n_max(0), perctrain(dsp.perctrain), perctest(dsp.perctest), splits(dsp.splits), _randomize(dsp._randomize) {current_split=dsp.current_split; notify("Data_Splitter_RandomRandom copy constructor.");}
	virtual ~Data_Splitter_RandomRandom() {notify("Data_Splitter_RandomRandom destructor.");}
	virtual IDXTYPE getNoOfSplits() const {assert(splits>0); return splits;}
	virtual bool makeFirstSplit();
	virtual bool makeNextSplit();
	Data_Splitter_RandomRandom* stateless_clone() const {return new Data_Splitter_RandomRandom(*this);}
	virtual std::ostream& print(std::ostream& os) const {os << "Data_Splitter_RandomRandom(splits=" << splits << ", %train=" << perctrain << ", %test=" << perctest << ")"; return os;}
protected:
	typedef Data_Splitter<INTERVALCONTAINER,IDXTYPE> TCC;
	virtual void makeRandomSplit(const IDXTYPE n, const boost::shared_ptr<INTERVALCONTAINER> list_train, const boost::shared_ptr<INTERVALCONTAINER> list_test);

	void fill_randomly(const IDXTYPE n, const BINTYPE id_empty, const BINTYPE id_fill, const IDXTYPE count, const IDXTYPE minidx, const IDXTYPE maxidx);
	void fill(const IDXTYPE n, const BINTYPE id_empty, const BINTYPE id_fill, const IDXTYPE minidx, const IDXTYPE maxidx);
	void translate(const IDXTYPE n, const BINTYPE id_fill, const boost::shared_ptr<INTERVALCONTAINER> lst);
	
	IDXTYPE _n_max; // actual _data size
	boost::scoped_array<BINTYPE> _data; // pattern selection array

	const IDXTYPE perctrain;
	const IDXTYPE perctest;
	const IDXTYPE splits;
	const bool _randomize;
	IDXTYPE current_split;
};

template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
void Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE>::fill_randomly(const IDXTYPE n, const BINTYPE id_empty, const BINTYPE id_fill, const IDXTYPE count, const IDXTYPE minidx, const IDXTYPE maxidx)
{
	assert(_data);
	assert(_n_max>=n);
	assert(minidx>=0);
	assert(maxidx>=minidx);
	assert(maxidx<n);
	assert(0<=count && count<=maxidx-minidx+1);
	// NOTE: assumes at least count id_empty slots exist in _data in interval <minidx,maxidx>
#ifdef DEBUG
	IDXTYPE dbg_cnt=0;
	for(IDXTYPE dbg_i=minidx;dbg_i<maxidx;dbg_i++) if(_data[dbg_i]==id_empty) ++dbg_cnt;
	assert(count<=dbg_cnt);
#endif
	IDXTYPE piv;
	for(IDXTYPE i=0;i<count;i++) // randomly choose samples
	{ 
		piv=minidx+(IDXTYPE)(rand()%(int)(maxidx-minidx+1)); assert(minidx<=piv && piv<=maxidx);
		while(_data[piv]!=id_empty) {piv++; if(piv>maxidx) piv=minidx;}
		_data[piv]=id_fill;
	}
}

template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
void Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE>::fill(const IDXTYPE n, const BINTYPE id_empty, const BINTYPE id_fill, const IDXTYPE minidx, const IDXTYPE maxidx)
{
	assert(_data);
	assert(_n_max>=n);
	assert(minidx>=0);
	assert(maxidx>=minidx);
	assert(maxidx<n);
	for(IDXTYPE i=minidx;i<=maxidx;i++) if(_data[i]==id_empty) _data[i]=id_fill;
}

template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
void Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE>::translate(const IDXTYPE n, const BINTYPE id_fill, const boost::shared_ptr<INTERVALCONTAINER> lst)
{
	assert(lst);
	lst->clear();
	IDXTYPE piv=0, count=0;
	while(piv<n && _data[piv]!=id_fill) piv++;
	while(piv<n)
	{
		count=1;
		while(piv+count<n && _data[piv+count]==id_fill) ++count;
		Data_Interval<IDXTYPE> tin={piv,count};
		lst->push_back(tin);
		piv+=count;
		while(piv<n && _data[piv]!=id_fill) piv++;
	}
}

template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
void Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE>::makeRandomSplit(const IDXTYPE n, const boost::shared_ptr<INTERVALCONTAINER> list_train, const boost::shared_ptr<INTERVALCONTAINER> list_test)
{
	try {
		assert(list_train);
		assert(list_test);
		assert(n>0);
		assert(0<=perctrain); // better (0<perctrain) ?
		assert(0<=perctest); // better (0<perctest) ?
		assert(perctrain+perctest<=100);
		if(!_data || _n_max<n) {_data.reset(new BINTYPE[n]); _n_max=n;}
		const BINTYPE id_empty = 0;
		const BINTYPE id_train = 1;
		const BINTYPE id_test = 2;
		const IDXTYPE trsiz=(n * perctrain)/100;
		const IDXTYPE tesiz=(n * perctest)/100;
		assert(trsiz+tesiz<=n);
		IDXTYPE i;
		for(i=0;i<n;i++) _data[i]=id_empty;

		// now fill randomly perctrain% for training and perctest% for testing
		// NOTE: acceleration - rand() needs to be called only min{perctrain,100-perctrain} and min{perctest,100-perctest} times - just adjust _data pre-fill
		//                    - also, if(perctrain+perctest==100), then rand() loop is needed only for one of them
		if(perctrain<=50) {
			fill_randomly(n,id_empty,id_train,trsiz/*count*/,0/*minidx*/,n-1/*maxidx*/); // randomly choose train samples
		} else {
			fill(n,id_empty,id_train,0/*minidx*/,n-1/*maxidx*/);
			fill_randomly(n,id_train,id_empty,n-trsiz/*count*/,0/*minidx*/,n-1/*maxidx*/); // randomly choose non-train samples
		}
		if(perctrain+perctest==100) {
			fill(n,id_empty,id_test,0/*minidx*/,n-1/*maxidx*/);
		} else if(perctest<=(100-perctrain)/2) {
			fill_randomly(n,id_empty,id_test,tesiz/*count*/,0/*minidx*/,n-1/*maxidx*/); // randomly choose test samples
		} else {
			fill(n,id_empty,id_test,0/*minidx*/,n-1/*maxidx*/);
			fill_randomly(n,id_test,id_empty,n-trsiz-tesiz/*count*/,0/*minidx*/,n-1/*maxidx*/); // randomly choose non-test samples
		}

		// now transform the binary representation to TCC::train and TCC::test lists
		translate(n,id_train,list_train);
		translate(n,id_test,list_test);
		
	} catch(...) {
		throw fst_error("Data_Splitter_RandomRandom::makeRandomSplit() error.");
	}
}

template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
bool Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE>::makeFirstSplit()
{
	TCC::assert_csplit();
	try {
		if(_randomize) srand( (unsigned int)time(NULL) );
		for(typename TCC::ClassIterator c=TCC::csplit.begin();c!=TCC::csplit.end();c++) {
			makeRandomSplit((*c)->n,(*c)->train,(*c)->test);
		}
		current_split=1;
	} catch(...) {
		throw fst_error("Data_Splitter_RandomRandom::makeFirstSplit() error.");
	}
	return true;
}

template<class INTERVALCONTAINER, typename IDXTYPE, typename BINTYPE>
bool Data_Splitter_RandomRandom<INTERVALCONTAINER,IDXTYPE,BINTYPE>::makeNextSplit()
{
	try {
		if(current_split<splits) {
			++current_split;
			for(typename TCC::ClassIterator c=TCC::csplit.begin();c!=TCC::csplit.end();c++) {
				makeRandomSplit((*c)->n,(*c)->train,(*c)->test);
			}
			return true;
		} else current_split=0;
	} catch(...) {
		throw fst_error("Data_Splitter_RandomRandom::makeNextSplit() error.");
	}
	return false;
}


} // namespace
#endif // FSTDATASPLITTERRANDRAND_H ///:~
