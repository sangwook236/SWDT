#ifndef FSTDATAACCESSORSPLITTINGMEM_H
#define FSTDATAACCESSORSPLITTINGMEM_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    data_accessor_splitting_mem.hpp
   \brief   Implements data access to data cached entirely in memory, concrete file type support is delegated to derived classes
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
#include "data_accessor_splitting.hpp"

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

//! Implements data access to data cached entirely in memory, concrete file type support is delegated to derived classes
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
class Data_Accessor_Splitting_Mem : public Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>
{
	// NOTE: this modified version unifies calling conventions for both inner and outer CV loops
	// - in both loops the same methods are to be called (just get*Split, get*Block, get*Pattern), while
	//   the actual inner/outer mode is to be switched on/off by 
	public:
		typedef Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER> DAS;
		typedef boost::shared_ptr<Data_Scaler<DATATYPE> > PScaler;
		typedef typename DAS::PPattern PPattern;
		typedef typename DAS::PSplitters PSplitters;
		Data_Accessor_Splitting_Mem(const string _filename, const PSplitters _dsp, const PScaler _dsc) : Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>(_dsp), dsc(_dsc), filename(_filename), datasize(0) {notify("Data_Accessor_Splitting_Mem constructor.");}
		virtual ~Data_Accessor_Splitting_Mem(){notify("Data_Accessor_Splitting_Mem destructor.");}
		
		//! to be called after object creation (for initialization of data file access etc.)
		virtual void initialize(); //!< \note off-limits in shared_clone
		
		virtual bool getFirstBlock(const DataPart ofwhat, PPattern &firstpattern, IDXTYPE &patterns, const unsigned int loopdepth=0);
		virtual bool getNextBlock(const DataPart ofwhat, PPattern &firstpattern, IDXTYPE &patterns, const unsigned int loopdepth=0);

		Data_Accessor_Splitting_Mem* clone() const {throw fst_error("Data_Accessor_Splitting_Mem::clone() not supported, use Data_Accessor_Splitting_Mem::sharing_clone() instead.");}
		Data_Accessor_Splitting_Mem* sharing_clone() const = 0;
		Data_Accessor_Splitting_Mem* stateless_clone() const {throw fst_error("Data_Accessor_Splitting_Mem::stateless_clone() not supported, use Data_Accessor_Splitting_Mem::sharing_clone() instead.");}
		
		virtual std::ostream& print(std::ostream& os) const;
	protected:
		Data_Accessor_Splitting_Mem(const Data_Accessor_Splitting_Mem &damt, int); // weak (referencing) copy-constructor to be used in sharing_clone()
	protected:
		virtual void initial_data_read() = 0;    //!< \note off-limits in shared_clone
		virtual void initial_file_prepare() = 0;
		virtual unsigned int file_getNoOfClasses() const = 0;
		virtual unsigned int file_getNoOfFeatures() const = 0;
		virtual IDXTYPE file_getClassSize(unsigned int cls) const = 0;

		void initial_data_scaling(); //!< \note off-limits in shared_clone
		
		PScaler dsc; //!< \note off-limits in shared_clone

		string filename; //!< \note off-limits in shared_clone
		
		IDXTYPE datasize;
		boost::shared_array<DATATYPE> data; //!< \note read-only in shared_clone
		//! class start indexes (offsets) in whatever random-access data representation
		vector<IDXTYPE> _class_offset; //!< \note read-only in shared_clone
};

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initial_data_scaling() //!< \note off-limits in shared_clone
{
	if(Clonable::is_sharing_clone()) throw fst_error("Data_Accessor_Splitting_Mem()::initial_data_scaling() called from shared_clone instance.");
	// SCALING over values per-feature, regardless classes
	assert(dsc);
	{std::ostringstream sos; sos << std::endl << "Scaling data (per-feature, over all classes) using " << *dsc << "..."; syncout::print(std::cout,sos);} //using " << *dsc << "...";}
	IDXTYPE idx=0;
	IDXTYPE _features=file_getNoOfFeatures();
	for(IDXTYPE f=0;f<_features;f++) {
		for(bool b=dsc->startFirstLoop(); b==true; b=dsc->startNextLoop())
		{
			idx=f;
			for(unsigned int c=0;c<file_getNoOfClasses();c++) for(IDXTYPE p=0;p<file_getClassSize(c);p++)
			{
				dsc->learn(data[idx]);
				idx+=_features;
			}
		}
		idx=f;
		for(unsigned int c=0;c<file_getNoOfClasses();c++) for(IDXTYPE p=0;p<file_getClassSize(c);p++)
		{
			dsc->scale_inplace(data[idx]);
			idx+=_features;
		}
		if(false) {std::ostringstream sos; sos << "scaled feature " << f << ": " << *dsc << std::endl; syncout::print(std::cout,sos);}
	}
	{std::ostringstream sos; sos << "done." << std::endl << std::endl; syncout::print(std::cout,sos);}
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initialize() //!< \note off-limits in shared_clone
{
	// NOTE: throws I/O and memory related exceptions
	if(Clonable::is_sharing_clone()) throw fst_error("Data_Accessor_Splitting_Mem()::initialize() called from shared_clone instance.");
	
	assert(DAS::dsp->size()>0);
	try {
		{std::ostringstream sos; sos << "Opening file: " << filename << std::endl; syncout::print(std::cout,sos);}
		initial_file_prepare();
		{
			std::ostringstream sos;
			sos << "Features: " << file_getNoOfFeatures() << std::endl << "Classes: " << file_getNoOfClasses() << std::endl;
			syncout::print(std::cout,sos);
		}
			
		typename DAS::CLASSSIZES csizes;
		for(unsigned int c=0;c<file_getNoOfClasses();c++) 
		{
			{std::ostringstream sos; sos << "  Class "<<c+1<<" size: "<<file_getClassSize(c) << std::endl; syncout::print(std::cout,sos);}
			csizes.push_back(file_getClassSize(c));
		}
		{
			std::ostringstream sos;
			sos << "Splitting depth: " << DAS::dsp->size() << std::endl;
			syncout::print(std::cout,sos);
		}
		
		DAS::initialize(file_getNoOfFeatures(), csizes); // set-up memory structures to support data splitting
		
		datasize=0;
		_class_offset.clear();
		for(unsigned int c=0;c<file_getNoOfClasses();c++) {
			_class_offset.push_back(datasize); // to be set only once
			datasize+=file_getClassSize(c)*file_getNoOfFeatures();
		}

		data.reset(new DATATYPE[datasize]);
		initial_data_read();
		initial_data_scaling();

	} catch (...) {
		notify("Data_Accessor_Splitting_Mem::initialize() failed.");
		throw;
	}
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
bool Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getFirstBlock(const DataPart ofwhat, PPattern &firstpattern, IDXTYPE &patterns, const unsigned int loopdepth)
{
	assert(data);
	typename DAS::DATAINTERVAL tmp;
	if(!DAS::getFirstBlock(ofwhat,tmp,loopdepth)) return false;
	firstpattern=&data[_class_offset[DAS::active_class]+tmp->startidx*DAS::features];
	patterns=tmp->count;
	return true;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
bool Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNextBlock(const DataPart ofwhat, PPattern &firstpattern, IDXTYPE &patterns, const unsigned int loopdepth)
{
	assert(data);
	typename DAS::DATAINTERVAL tmp;
	if(!DAS::getNextBlock(ofwhat,tmp,loopdepth)) return false;
	firstpattern=&data[_class_offset[DAS::active_class]+tmp->startidx*DAS::features];
	patterns=tmp->count;
	return true;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::Data_Accessor_Splitting_Mem(const Data_Accessor_Splitting_Mem& damt, int) : Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>(damt),
	datasize(damt.datasize),
	data(damt.data),
	_class_offset(damt._class_offset)
{
	dsc.reset();
	filename.clear();
	notify("Data_Accessor_Splitting_Mem weak copy-constructor.");
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
std::ostream& Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>::print(std::ostream& os) const 
{
	DAS::print(os); 
	os << std::endl << "Data_Accessor_Splitting_Mem(file=" << filename << ")";
	return os;
}
	
} // namespace
#endif // FSTDATAACCESSORSPLITTINGMEM_H ///:~
