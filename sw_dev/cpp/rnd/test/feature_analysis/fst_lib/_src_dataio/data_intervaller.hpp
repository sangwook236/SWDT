#ifndef FSTDATAINTERVALLER_H
#define FSTDATAINTERVALLER_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    data_intervaller.hpp
   \brief   Container to hold a list of Data_Interval; implements nested interval reduction
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
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector> // for blockiter - array of iterators indexed by loopdepth
#include "error.hpp"
#include "global.hpp"

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

//! Support structure to hold data interval (referring into data held by data accessor)
template<typename IDXTYPE>
struct Data_Interval {
	IDXTYPE startidx;
	IDXTYPE count;
};

template <class CONTAINER, typename IDXTYPE>
class Data_Intervaller;

template <class CONTAINER, typename IDXTYPE>
std::ostream& operator<<(std::ostream& os, const Data_Intervaller<CONTAINER,IDXTYPE>& ia);

//! Container to hold list of Data_Interval; implements nested interval reduction
template<class CONTAINER, typename IDXTYPE>
//! \invariant CONTAINER must be a container of Data_Interval supporting iterators, push_back(), clear()
class Data_Intervaller : public CONTAINER {
public:
	Data_Intervaller() {notify("Data_Intervaller constructor.");};
	Data_Intervaller(const Data_Intervaller& tin);
	virtual ~Data_Intervaller() {notify("Data_Intervaller destructor.");};
	Data_Interval<IDXTYPE>* const getFirstBlock(const unsigned int loopdepth=0);
	Data_Interval<IDXTYPE>* const getNextBlock(const unsigned int loopdepth=0);
	bool check_validity(const IDXTYPE maxidx);		//!< checks whether all intervals are non-overlapping, ordered ascending, and all fit to <0,maxidx> (maxidx==0 disables maxidx test)
	IDXTYPE sum();	//!< sums the sizes of all contained intervals
	IDXTYPE max_idx();	//!< returns maximum of all startidx+count values
	void reduce(const boost::shared_ptr<Data_Intervaller<CONTAINER,IDXTYPE> > reducer, boost::shared_ptr<Data_Intervaller<CONTAINER,IDXTYPE> > target);	//!< splits the "mass" contained here in intervals according to the "reducer" and store the reduced result in target
	friend std::ostream& operator<< <>(std::ostream& os, const Data_Intervaller<CONTAINER,IDXTYPE>& ia);
private:
	typedef typename Data_Intervaller<CONTAINER,IDXTYPE>::iterator TCI;
	std::vector<TCI> blockiter; //!< to enable multiple independent get*Block() loops
};

template<class CONTAINER, typename IDXTYPE>
Data_Intervaller<CONTAINER,IDXTYPE>::Data_Intervaller(const Data_Intervaller& tin)
{
	notify("Data_Intervaller copy constructor.");
	this->clear();
	for(typename Data_Intervaller<CONTAINER,IDXTYPE>::const_iterator iter=tin.begin();iter!=tin.end();iter++) this->push_back(*iter);
	for(typename std::vector<TCI>::const_iterator biter=tin.blockiter.begin();biter!=tin.blockiter.end();biter++) this->blockiter.push_back(*biter);
}

template<class CONTAINER, typename IDXTYPE>
std::ostream& operator<<(std::ostream& os, const Data_Intervaller<CONTAINER,IDXTYPE>& ia) {
	for(typename Data_Intervaller<CONTAINER,IDXTYPE>::const_iterator iter=ia.begin();iter!=ia.end();iter++) os << "si:" << iter->startidx << ", c:" << iter->count << std::endl;
	return os;
}

template<class CONTAINER, typename IDXTYPE>
Data_Interval<IDXTYPE>* const Data_Intervaller<CONTAINER,IDXTYPE>::getFirstBlock(const unsigned int loopdepth)
{
	if(blockiter.size()<=loopdepth) blockiter.resize(loopdepth+1);
	blockiter[loopdepth]=this->begin();
	if(blockiter[loopdepth]!=this->end()) return &(*blockiter[loopdepth]);
	else return NULL;
}

template<class CONTAINER, typename IDXTYPE>
Data_Interval<IDXTYPE>* const Data_Intervaller<CONTAINER,IDXTYPE>::getNextBlock(const unsigned int loopdepth)
{
	// NOTE: getNextBlock() must be preceeded by getNextBlock() with the same loopdepth value, othervise the behaviour is undefined
	assert(loopdepth<blockiter.size());
	++blockiter[loopdepth];
	if(blockiter[loopdepth]!=this->end()) return &(*blockiter[loopdepth]);
	else return NULL;
}

template<class CONTAINER, typename IDXTYPE>
bool Data_Intervaller<CONTAINER,IDXTYPE>::check_validity(const IDXTYPE maxidx)
{
	if(this->empty()) {{std::ostringstream sos; sos << "Data_Intervaller is empty" << std::endl; syncout::print(std::cerr,sos);} return false;}
	TCI iter=this->begin();
	IDXTYPE pivot=iter->startidx+iter->count;
	if(iter->startidx<0 || (maxidx>0 && (pivot>maxidx)) ) {{std::ostringstream sos; sos <<"first is wrong"<<std::endl; syncout::print(std::cerr,sos);} return false;} // the first interval reaches outside <0,maxidx>
	if(iter==this->end()) return true; //just one interval, and that one is OK
	iter++;
	while(iter!=this->end())
	{
		if(iter->startidx<pivot) {{std::ostringstream sos; sos <<"iter->startidx="<<iter->startidx<<", pivot="<<pivot<<std::endl; syncout::print(std::cerr,sos);} return false;}
		pivot=iter->startidx+iter->count;
		if(maxidx>0 && (pivot>maxidx)) {{std::ostringstream sos; sos <<"startidx+count>maxidx"<<std::endl; syncout::print(std::cerr,sos);} return false;}
		++iter;
	}
	return true;
}

template<class CONTAINER, typename IDXTYPE>
IDXTYPE Data_Intervaller<CONTAINER,IDXTYPE>::sum()
{
	IDXTYPE cnt=0;
	for(TCI iter=this->begin();iter!=this->end();iter++) cnt+=iter->count;
	return cnt;
}

template<class CONTAINER, typename IDXTYPE>
IDXTYPE Data_Intervaller<CONTAINER,IDXTYPE>::max_idx()
{
	IDXTYPE mx=0;
	for(TCI iter=this->begin();iter!=this->end();iter++) if(iter->startidx+iter->count-1>mx) mx=iter->startidx+iter->count-1;
	return mx;
}

template<class CONTAINER, typename IDXTYPE>
void Data_Intervaller<CONTAINER,IDXTYPE>::reduce(const boost::shared_ptr<Data_Intervaller<CONTAINER,IDXTYPE> > reducer, boost::shared_ptr<Data_Intervaller<CONTAINER,IDXTYPE> > target)
{
	try{
		assert(sum()>=reducer->max_idx());
		target->clear();
		typename Data_Intervaller<CONTAINER,IDXTYPE>::const_iterator rediter=reducer->begin();
		IDXTYPE Mbottom=0;
		for(TCI iter=this->begin();iter!=this->end();iter++) {
			bool stop=false;
			while((!stop) && (rediter!=reducer->end())) {
				IDXTYPE Rbottom=(*rediter).startidx;
				IDXTYPE Rtop=Rbottom+(*rediter).count-1;
				IDXTYPE MTRUEbottom=(*iter).startidx;
				IDXTYPE Mtop=Mbottom+(*iter).count-1;
				if(!(Rtop<Mbottom || Rbottom>Mtop)) { //write intersection to target
					IDXTYPE ISCbottom=Rbottom; if(Rbottom<Mbottom) ISCbottom=Mbottom;
					IDXTYPE ISCtop=Rtop; if(Rtop>Mtop) ISCtop=Mtop;
					Data_Interval<IDXTYPE> I={MTRUEbottom+(ISCbottom-Mbottom), ISCtop-ISCbottom+1};
					target->push_back(I);
				}
				if(Rtop>=Mtop) stop=true;
				if(Rtop<=Mtop) rediter++;
			}
			Mbottom+=(*iter).count;
		}
		
	} catch(...) {
		throw fst_error("Data_Intervaller::reduce() error.");
	}
}


} // namespace
#endif // FSTDATAINTERVALLER_H ///:~
