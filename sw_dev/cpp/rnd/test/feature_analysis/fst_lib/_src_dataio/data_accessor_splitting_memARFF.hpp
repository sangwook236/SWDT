#ifndef FSTDATAACCESSORSPLITTINGMEMARFF_H
#define FSTDATAACCESSORSPLITTINGMEMARFF_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    data_accessor_splitting_memARFF.hpp
   \brief   Implements data access to data cached entirely in memory, read once from ARFF files
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at http://fst.utia.cz
   \date    November 2010
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
#include <sstream>
#include <vector>

#include "error.hpp"
#include "global.hpp"
#include "data_accessor_splitting_mem.hpp"
#include "data_file_ARFF.hpp"

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

//! Implements data access to data cached entirely in memory, read once from a ARFF file
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
class Data_Accessor_Splitting_MemARFF : public Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>
{
	// NOTE: this modified version unifies calling conventions for both inner and outer CV loops
	// - in both loops the same methods are to be called (just get*Split, get*Block, get*Pattern), while
	//   the actual inner/outer mode is to be switched on/off by 
	public:
		typedef Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER> DASM;
		typedef boost::shared_ptr<Data_Scaler<DATATYPE> > PScaler;
		typedef typename DASM::PSplitters PSplitters;
		Data_Accessor_Splitting_MemARFF(const string _filename, const PSplitters _dsp, const PScaler _dsc) : Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>(_filename, _dsp, _dsc) {notify("Data_Accessor_Splitting_MemARFF constructor.");}
		virtual ~Data_Accessor_Splitting_MemARFF(){notify("Data_Accessor_Splitting_MemARFF destructor.");}
		
		Data_Accessor_Splitting_MemARFF* sharing_clone() const;
		
		virtual std::ostream& print(std::ostream& os) const;
	protected:
		Data_Accessor_Splitting_MemARFF(const Data_Accessor_Splitting_MemARFF &damt, int x) : Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>(damt, x) {}; // weak (referencing) copy-constructor to be used in sharing_clone()
	protected:
		boost::scoped_ptr<Data_File_ARFF> ARFFFile;

		unsigned int clsCount, featureCount;
		boost::shared_array<IDXTYPE> clsSize;

		virtual void initial_data_read() ;    //!< \note off-limits in shared_clone
		virtual void initial_file_prepare();
		
    public:
		virtual unsigned int file_getNoOfClasses() const {return clsCount; };
		virtual unsigned int file_getNoOfFeatures() const {return featureCount; };
		virtual IDXTYPE file_getClassSize(unsigned int cls) const {return clsSize[cls]; };
};

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initial_file_prepare() //!< \note off-limits in shared_clone
{
	if(Clonable::is_sharing_clone()) throw fst_error("Data_Accessor_Splitting_MemARFF()::initial_data_prepare() called from shared_clone instance.");
	ARFFFile.reset(new Data_File_ARFF(this->filename));
	ARFFFile->prepareFile();
	ARFFFile->sortRecordsByClass();
	
	clsCount = ARFFFile->getNoOfClasses();
	featureCount = ARFFFile->getNoOfFeatures();
	clsSize.reset(new IDXTYPE[clsCount]);
	for (unsigned int i=0;i<clsCount; i++) clsSize[i] = ARFFFile->getClassSize(i);
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initial_data_read() //!< \note off-limits in shared_clone
{
	if(Clonable::is_sharing_clone()) throw fst_error("Data_Accessor_Splitting_MemARFF()::initial_data_read() called from shared_clone instance.");
	{std::ostringstream sos; sos << "Reading data to memory..."; syncout::print(std::cout,sos);}
	IDXTYPE idx=0;
	for(unsigned int p=0;p<ARFFFile->getTotalSize();p++) 
	{
		for(unsigned int f=0;f<ARFFFile->getNoOfFeatures();f++) {
			this->data[idx++]=(DATATYPE) ARFFFile->getFeatureValue(p,f); 
		}
	}
	ARFFFile.reset();
	{std::ostringstream sos; sos << "done."<<endl; syncout::print(std::cout,sos);}
}

/*! \warning A sharing_clone() object shares 'data' with the original object. Although the sharing_clone
             object can not modify 'data', there is no synchronization implemented to avoid
             concurrent modification of 'data' by the original object while reading it from the sharing_clone! */
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALCONTAINER>* Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALCONTAINER>::sharing_clone() const
{
	Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALCONTAINER> *clone=new Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALCONTAINER>(*this, (int)0);
	clone->set_sharing_cloned();
	return clone;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
std::ostream& Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALCONTAINER>::print(std::ostream& os) const 
{
	DASM::print(os); 
	os << std::endl << "Data_Accessor_Splitting_MemARFF()";
	return os;
}
	
} // namespace
#endif // FSTDATAACCESSORSPLITTINGMEMARFF_H ///:~
