#ifndef FSTDATAACCESSORSPLITTING_H
#define FSTDATAACCESSORSPLITTING_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    data_accessor_splitting.hpp
   \brief   Defines support for data splitting
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
#include <vector>
#include "error.hpp"
#include "global.hpp"
#include "data_accessor.hpp"
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

/*! \brief partly-abstract class, defines support for data splitting

\detailed Data structures in Data_Accessor_Splitting are directly used in
the splitting mechanism that enables structured access to data.
In order to keep the implementation of data splitters
(specializations of Data_Splitter) as simple as possible
for the user, we moved most of the technicalities here.
Correct state of key data structures in Data_Accessor_Splitting is as follows:

- dsp represents a list of Data_Splitters while each splitter keeps
      for each class two pointers: a pointer to the list of train and test data (lists of data intervals).
      The splitters are, however, not the owners of the actual train and test lists,
      these are allocated within Data_Accessor_Splitting.
      
The splitting mechanism needs not just one train-test data structure pair,
but two - the second pair denoted _reduced_train and _reduced_test. The
"reduced" pair is constructed from the "base" pair separately in each
splitting level, so as to correctly represent the subset of data defined by
the respective splitter (each deeper level further reduces access to data
visible in the preceeding level). The actual access to data through
getFirstBlock() and getNextBlock() is commanded by data intervals stored in
the "reduced" pair of interval lists only.

The two pairs of interval lists exist separately for each splitting level
and each data class. In Data_Accessor_Splitting they are collected in the DataSplit
subclass, of which the required number of instances is kept in the "splits"
container. In correct representation "splits" must contain 
[number of classes]*[number of splitting levels] DataSplit instances.

The DataSplit that represent top splitting level differ from the deeper
level - the "reduced" pair of lists is actually just referencing the "base"
pair. This is because data indexes as produced by splitters are valid
indexes usable to access the data. In deeper splitting levels this is not so
because splitter indexes must be treated are relative to the data possibly restricted
in higher level. Transforming the relative indexes to absolute indexes is
achieved through the "reduce" method implemented in Data_Intervaller.
In non-top splitting levels before data can be accessed, the "base" indexes/intervals
are first transformed using the "reduce" method with the result stored
in "_reduced" train and test lists.

The "base" train and test lists allocated here in Data_Accessor_Splitting need to
be interlinked with respective data splitters. The splitters do not hold
any allocated structures, they re-direct their output to the "base"
train and test lists kept in Data_Accessor_Splitting, to enable data accessing
routines to transform the indexes by means of "reduce" whenever needed
and subsequently to access the correct subset of data.
*/
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
class Data_Accessor_Splitting : public Data_Accessor<DATATYPE,IDXTYPE>
{
	// NOTE: this modified version unifies calling conventions for nested data access loops (like in cross-validation)
	//   in all loops the same methods are to be called (just get*Split, get*Block, get*Pattern), while
	//   the actual loop level is to be set using setSplittingDepth()
	// Access to data is defined using indexes, assuming data is accessible by random access
	// Support for arbitrary data splitting (train/test, cross-validation, hold-out, etc., by means
	// of arbitrary Data_Splitter objects) is implemented here
	public:
		typedef boost::shared_ptr<Data_Splitter<INTERVALCONTAINER,IDXTYPE> > PSplitter;
		typedef boost::shared_ptr<std::vector<PSplitter> > PSplitters;
		typedef const DATATYPE* PPattern;
		Data_Accessor_Splitting(const PSplitters _dsp); //, const PScaler _dsc);
		virtual ~Data_Accessor_Splitting(){notify("Data_Accessor_Splitting destructor.");}	
		
		virtual unsigned int getNoOfClasses() const;          //!< returns number of classes
		virtual unsigned int getNoOfFeatures() const;         //!< returns data dimensionality
		
		virtual IDXTYPE getClassSize(const unsigned int c) const;    //!< returns size (number of samples in) of class c
		virtual IDXTYPE getClassSizeSum() const;    //!< returns summed size (number of samples in) of all classes, i.e., no. of all patterns in data
		
		virtual void setClass(const int c) {assert(c>=0); active_class = c;} //!< sets active class -> from now on only data from class c will be considered
		virtual int  getClass() const {return active_class;}

		        void setSplittingDepth(const unsigned int depth);
		        unsigned int getSplittingDepth() const {return splitting_depth;}
		        
		virtual unsigned int getNoOfSplits() const;  //!< data access iteration (to support, e.g., loops in cross-validation)
		virtual bool getFirstSplit();                //!< data access iteration (to support, e.g., loops in cross-validation)
		virtual bool getNextSplit();                 //!< data access iteration (to support, e.g., loops in cross-validation)
		virtual unsigned int getSplitIndex() const;  //!< data access iteration (to support, e.g., loops in cross-validation)

		/*! \note in block accessing methods use different loopdepth whenever two or more loops (of any DataPart type) should overlap, 
		          otherwise the behaviour is undefined */
		
		//! \warning getNoOfBlocks is not to be relied upon due to possible limitations in some implementations
		virtual IDXTYPE getNoOfBlocks(const DataPart ofwhat) const;
		//! returns pointer to first consecutive block of data of requested DataPart type in the current split (access iteration)
		virtual bool getFirstBlock(const DataPart ofwhat, PPattern &firstpattern, IDXTYPE &patterns, const unsigned int loopdepth=0)=0;
		//! returns pointer to next consecutive block of data of requested DataPart type in the current split (access iteration)
		virtual bool getNextBlock(const DataPart ofwhat, PPattern &firstpattern, IDXTYPE &patterns, const unsigned int loopdepth=0)=0;
		//! returns index of the current consecutive block of data of requested DataPart type in the current split (access iteration)
		virtual IDXTYPE getBlockIndex(const unsigned int loopdepth=0) const;
		//! returns number of patterns in all consecutive blocks of data of requested DataPart type in the current split (access iteration)
		virtual IDXTYPE getNoOfPatterns(const DataPart ofwhat) const;
		//! enables change of meaning of DataPart types, for use in specialized data access scenarios like in bias predicting wrappers
		virtual void substitute(const DataPart source, const DataPart target);
		//! resets standard DataPart types' meaning
		virtual void resubstitute();

		virtual std::ostream& print(std::ostream& os) const;
	protected:
		Data_Accessor_Splitting(const Data_Accessor_Splitting &da);      // to be used in clone() in derived classes
	protected:		
		typedef std::vector<unsigned int> CLASSSIZES;
		/*! sets-up memory structures needed in the splitting mechanism
		    data access is not needed here, the structures work with indexes
		    only - the only information needed is dimensionality and sizes of data classes */
		void initialize(const unsigned int _features, const CLASSSIZES &_classes); ////const unsigned int _features, 
		CLASSSIZES classes;
		unsigned int features;

		// DataPart mapping
		DataPart mappedTRAIN, mappedTEST, mappedTRAINTEST, mappedALL;
		DataPart mappedDataPart(const DataPart ofwhat) const;

		// splitting support
		PSplitters dsp;
		typedef const Data_Interval<IDXTYPE>* DATAINTERVAL;
		//! returns Data_Interval record representing the first consecutive block of data of requested DataPart type in the current split (access iteration)
		virtual bool getFirstBlock(const DataPart ofwhat, DATAINTERVAL &tmp, const unsigned int loopdepth=0);
		//! returns Data_Interval record representing the next consecutive block of data of requested DataPart type in the current split (access iteration)
		virtual bool getNextBlock(const DataPart ofwhat, DATAINTERVAL &tmp, const unsigned int loopdepth=0);

		typedef boost::shared_ptr<INTERVALCONTAINER> PIntervaller;
		//! Data splitting support structure; holds one set of intervals (train, test) per each splitting depth and (data)class
		class DataSplit {
		public:
			DataSplit() {_train.reset(); _test.reset(); _reduced_train.reset(); _reduced_test.reset(); notify("DataSplit empty constructor.");}
			DataSplit(const DataSplit& cs);      // "strong" cc (for use in clone)
			DataSplit(const DataSplit& cs, int); // "weak" cc (for use in shared_clone)
			~DataSplit() {notify("DataSplit destructor.");}
			PIntervaller _train;
			PIntervaller _test;
			PIntervaller _reduced_train;
			PIntervaller _reduced_test;
			Data_Interval<IDXTYPE> _all;   //!< to enable passing of block reference when DataPart ofwhat==ALL
		};
		std::vector<std::vector<DataSplit> > splits;    //!< one set of splitters per each splitting depth and class

		std::vector<IDXTYPE> enum_split;           //!< current split..  0~none
		std::vector<std::vector<IDXTYPE> > enum_block;  //!< current block loop..  0~none
		std::vector<std::vector<DataPart> > tt_phase;   //!< in current block loop.. for DataPart==TRAINTEST indicates: 0-no loop, 1-train loop, 2-test loop

		unsigned int splitting_depth;           //!< switch between inner and outer loop get*Train*, get*Test* functionality

		int active_class;                     //!< denotes from which class the get*Train* get*Validate* get*Test* methods return patterns, to be set using setClass()
		
		// for debugging
		bool _initialize_called;
		bool is_initialized() const {return _initialize_called;}
		void assert_splits(const int splitting_check=-1) const;
};


// ============================================ PUBLIC ===============================================

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::Data_Accessor_Splitting(const PSplitters _dsp) : //, const PScaler _dsc) : dsc(_dsc) 
	mappedTRAIN(TRAIN), mappedTEST(TEST), mappedTRAINTEST(TRAINTEST), mappedALL(ALL)
{
	notify("Data_Accessor_Splitting constructor.");
	_initialize_called=false;
	features=0;
	assert(_dsp->size()>0);
	dsp=_dsp;
	active_class=-1; 
	splitting_depth=0;
	splits.assign(_dsp->size(),std::vector<DataSplit>()); 
	enum_split.assign(_dsp->size(),0);
	enum_block.assign(_dsp->size(),std::vector<IDXTYPE>()); 
	tt_phase.assign(_dsp->size(),std::vector<DataPart>()); 
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
unsigned int Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNoOfClasses() const
{
	return classes.size();
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
unsigned int Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNoOfFeatures() const
{
	return features;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
IDXTYPE Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getClassSize(const unsigned int c) const
{
	assert(c>=0 && c<classes.size());
	return classes[c];
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
IDXTYPE Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getClassSizeSum() const
{
	IDXTYPE patterns=0;
	for(unsigned int c=0;c<classes.size();c++) patterns+=classes[c];
	return patterns;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::setSplittingDepth(const unsigned int depth) // false: get*Train* ~ Train+Validate, get*Test* ~ Test; true: get*Train* ~ Train, get*Test* ~ Validate;
{
#ifdef DEBUG
	assert(depth>=0 && depth<dsp->size());
	assert(enum_split.size()==dsp->size());
	for(unsigned int n=0;n<depth;n++) assert(enum_split[n]>0);
#endif
	splitting_depth=depth;
} 

// WARNING:  current implementation may return misleading results if _dsp_*_template objects hadn't been set before using set_n()
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
unsigned int Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNoOfSplits() const
{
	assert(dsp); //pridat kontrolu obsahu vektoru
	assert(splitting_depth>=0 && splitting_depth<dsp->size());
	return (*dsp)[splitting_depth]->getNoOfSplits();
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
bool Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getFirstSplit()
{
#ifdef DEBUG
	assert(dsp); //pridat kontrolu obsahu vektoru
	assert(is_initialized());
	assert(splitting_depth>=0 && splitting_depth<(unsigned int)dsp->size());
	for(unsigned int n=0;n<splitting_depth;n++) assert(enum_split[n]>0);
	assert_splits(/*n not set in splitters yet*/);
#endif
	bool result=true;
	if(splitting_depth>0)
		for(unsigned int c=0; c<classes.size(); c++) 
			(*dsp)[splitting_depth]->set_n(c, splits[splitting_depth][c]._all.count=splits[splitting_depth-1][c]._train->sum() );
	else
		for(unsigned int c=0; c<classes.size(); c++) 
			(*dsp)[splitting_depth]->set_n(c, splits[splitting_depth][c]._all.count=classes[c] );

	if(!(*dsp)[splitting_depth]->makeFirstSplit()) result=false;
	else 
	if(splitting_depth>0) for(unsigned int c=0; c<classes.size(); c++) {
		splits[splitting_depth-1][c]._train->reduce(splits[splitting_depth][c]._train,splits[splitting_depth][c]._reduced_train);
		splits[splitting_depth-1][c]._train->reduce(splits[splitting_depth][c]._test,splits[splitting_depth][c]._reduced_test);
	}
	if(result) enum_split[splitting_depth]=1; else enum_split[splitting_depth]=0;
	for(unsigned int n=splitting_depth+1;n<dsp->size();n++) enum_split[n]=0; // reset/cancel all deeper splits
	
	return result;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
bool Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNextSplit()
{
#ifdef DEBUG
	assert(dsp); //pridat kontrolu obsahu vektoru
	assert(is_initialized());
	assert(splitting_depth>=0 && splitting_depth<dsp->size());
	for(unsigned int n=0;n<=splitting_depth;n++) assert(enum_split[n]>0);
	assert_splits(splitting_depth);
#endif
	bool result=true;
	if(!(*dsp)[splitting_depth]->makeNextSplit()) result=false;
	else 
	if(splitting_depth>0) for(unsigned int c=0; c<classes.size(); c++) {
		splits[splitting_depth-1][c]._train->reduce(splits[splitting_depth][c]._train,splits[splitting_depth][c]._reduced_train);
		splits[splitting_depth-1][c]._train->reduce(splits[splitting_depth][c]._test,splits[splitting_depth][c]._reduced_test);
	}
	if(result) enum_split[splitting_depth]++; else enum_split[splitting_depth]=0;
	for(unsigned int n=splitting_depth+1;n<dsp->size();n++) enum_split[n]=0; // reset/cancel all deeper splits
		
	return result;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
unsigned int Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getSplitIndex() const
{
	assert(splitting_depth>=0 && splitting_depth<dsp->size());
	return enum_split[splitting_depth];
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
IDXTYPE Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNoOfBlocks(const DataPart ofwhat) const
{
	assert(dsp); //pridat kontrolu obsahu vektoru
	assert(is_initialized());
	assert(active_class>=0 && active_class<(int)classes.size());
	assert(splitting_depth>=0 && splitting_depth<dsp->size());
	assert_splits(splitting_depth);
	switch(mappedDataPart(ofwhat)) {
		case TRAIN: {return splits[splitting_depth][active_class]._reduced_train->size();}
		case TEST: {return splits[splitting_depth][active_class]._reduced_test->size();}
		case TRAINTEST: {return splits[splitting_depth][active_class]._reduced_train->size()+splits[splitting_depth][active_class]._reduced_test->size();}
		case ALL: {if(splitting_depth>0) return splits[splitting_depth-1][active_class]._reduced_train->size(); else return 1;}
		case NONE: {return 0;}
	} 
	return 0;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
IDXTYPE Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getBlockIndex(const unsigned int loopdepth) const
{
	assert(dsp); //pridat kontrolu obsahu vektoru
	assert(splitting_depth>=0 && splitting_depth<dsp->size());
	assert(loopdepth<enum_block[splitting_depth].size());
	return enum_block[splitting_depth][loopdepth];
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
IDXTYPE Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNoOfPatterns(const DataPart ofwhat) const
{
	assert(is_initialized());
	assert(splitting_depth>=0 && splitting_depth<dsp->size());
	assert(active_class>=0 && active_class<(int)classes.size());
	assert_splits(splitting_depth);
	switch(mappedDataPart(ofwhat)) {
		case TRAIN: {return splits[splitting_depth][active_class]._reduced_train->sum();}
		case TEST: {return splits[splitting_depth][active_class]._reduced_test->sum();}
		case TRAINTEST: {return splits[splitting_depth][active_class]._reduced_train->sum()+splits[splitting_depth][active_class]._reduced_test->sum();}
		case ALL: {if(splitting_depth>0) return splits[splitting_depth-1][active_class]._reduced_train->sum(); 
			else return classes[active_class];}
		case NONE: {return 0;}
	} 
	return 0;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::substitute(const DataPart source, const DataPart target)
{
	assert(source==TRAIN || source==TEST || source==TRAINTEST || source==ALL);
	assert(target==TRAIN || target==TEST || target==TRAINTEST || target==ALL);
	switch(source){
		case TRAIN:	mappedTRAIN=target; break;
		case TEST: mappedTEST=target; break;
		case TRAINTEST: mappedTRAINTEST=target; break;
		case ALL: mappedALL=target; break;
		case NONE: break;
	}
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::resubstitute()
{
	mappedTRAIN=TRAIN;
	mappedTEST=TEST;
	mappedTRAINTEST=TRAINTEST;
	mappedALL=ALL;
}


// ============================================ PROTECTED ===============================================

// strong copy-constructor, creates 1:1 copy of all structures
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::Data_Accessor_Splitting(const Data_Accessor_Splitting &da) :
	classes(da.classes),
	features(da.features),
	mappedTRAIN(da.mappedTRAIN), 
	mappedTEST(da.mappedTEST), 
	mappedTRAINTEST(da.mappedTRAINTEST), 
	mappedALL(da.mappedALL),
	//dsp(da.dsp),
	splits(da.splits),
	enum_split(da.enum_split),
	enum_block(da.enum_block),
	tt_phase(da.tt_phase),
	splitting_depth(da.splitting_depth),
	active_class(da.active_class),
	_initialize_called(da._initialize_called)
{
	notify("Data_Accessor_Splitting() copy constructor.");
	notify("DAS::copy-constructor, before da.assert_splits()");
	da.assert_splits();
	notify("DAS::copy-constructor, after da.assert_splits()");
	assert(da.dsp);
	assert(da.dsp->size()>0);
	dsp.reset(new std::vector<PSplitter>);
	for(unsigned int n=0; n<da.dsp->size();n++)
	{
		
		PSplitter ps(dynamic_cast<Data_Splitter<INTERVALCONTAINER,IDXTYPE>* >((*da.dsp)[n]->stateless_clone()));
		for(unsigned int c=0; c<classes.size(); c++)
			ps->assign(c,splits[n][c]._train,splits[n][c]._test);
		dsp->push_back(ps);
	}	
	notify("DAS::copy-constructor, before assert_splits()");
	assert_splits();
	notify("DAS::copy-constructor, after assert_splits()");
}


template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initialize(const unsigned int _features, const CLASSSIZES &_classes)
{
	notify("Data_Accessor_Splitting()::initialize.");
	if(is_initialized()) notify("Re-initializing Data_Accessor_Splitting.");
	classes.resize( _classes.size() );
	copy(_classes.begin(), _classes.end(), classes.begin() );
	features=_features;
	
	for(unsigned int n=0;n<dsp->size();n++) // number of nested splitting levels
	{
		splits[n].clear(); //splits[n].reserve(classes.size());
		notify("Data_Accessor_Splitting()::initialize to resize splits.");
		splits[n].resize(classes.size());
		notify("Data_Accessor_Splitting()::initialize splits resized.");
		for(unsigned int c=0;c<classes.size();c++) 
		{
			DataSplit &cs=splits[n][c];
			cs._train.reset(new INTERVALCONTAINER());
			cs._test.reset(new INTERVALCONTAINER());
			(*dsp)[n]->assign(c,cs._train,cs._test);
			if(n==0) {
				(*dsp)[0]->set_n(c,classes[c]);
				cs._reduced_train=cs._train;
				cs._reduced_test=cs._test;
			} else {
				cs._reduced_train.reset(new INTERVALCONTAINER());
				cs._reduced_test.reset(new INTERVALCONTAINER());
			}
			// NOTE: every time before use, each DataSplitter needs to be set to the valid number of samples..
		}
	}
	_initialize_called=true;
	notify("Data_Accessor_Splitting()::initialize finished.");
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
DataPart Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::mappedDataPart(const DataPart ofwhat) const
{
	switch(ofwhat){
		case TRAIN:	return mappedTRAIN;
		case TEST: return mappedTEST;
		case TRAINTEST: return mappedTRAINTEST;
		case ALL: return mappedALL;
		case NONE: return NONE;
	}
	return ofwhat;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
bool Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getFirstBlock(const DataPart ofwhat, DATAINTERVAL &tmp, const unsigned int loopdepth)
{
#ifdef DEBUG
	assert(dsp); //pridat kontrolu obsahu vektoru
	assert(is_initialized());
	assert(splitting_depth>=0 && splitting_depth<dsp->size());
	assert(active_class>=0 && active_class<(int)classes.size());
	for(unsigned int n=0;n<=splitting_depth;n++) assert(enum_split[n]>0);
	assert_splits(splitting_depth);
#endif
	if(enum_block[splitting_depth].size()<=loopdepth) enum_block[splitting_depth].resize(loopdepth+1);
	switch(mappedDataPart(ofwhat)) {
		case TRAIN: {tmp=splits[splitting_depth][active_class]._reduced_train->getFirstBlock(loopdepth); break;}
		case TEST: {tmp=splits[splitting_depth][active_class]._reduced_test->getFirstBlock(loopdepth); break;}
		case TRAINTEST: {
			if(tt_phase[splitting_depth].size()<=loopdepth) tt_phase[splitting_depth].resize(loopdepth+1);
			tt_phase[splitting_depth][loopdepth]=TRAIN; 
			tmp=splits[splitting_depth][active_class]._reduced_train->getFirstBlock(loopdepth); 
			if(tmp==NULL) {
				tt_phase[splitting_depth][loopdepth]=TEST; 
				tmp=splits[splitting_depth][active_class]._reduced_test->getFirstBlock(loopdepth); 
			}
			if(tmp==NULL) tt_phase[splitting_depth][loopdepth]=NONE;
			break;}
		case ALL: {if(splitting_depth>0) tmp=splits[splitting_depth-1][active_class]._reduced_train->getFirstBlock(loopdepth); 
			else tmp=&(splits[0][active_class]._all); break;}
		case NONE: {tmp=NULL;}
	}
	if(tmp!=NULL) enum_block[splitting_depth][loopdepth]=1; else {enum_block[splitting_depth][loopdepth]=0; return false;}
	return true;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
bool Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::getNextBlock(const DataPart ofwhat, DATAINTERVAL &tmp, const unsigned int loopdepth)
{
#ifdef DEBUG
	assert(dsp); //pridat kontrolu obsahu vektoru
	assert(is_initialized());
	assert(splitting_depth>=0 && splitting_depth<dsp->size());
	assert(active_class>=0 && active_class<(int)classes.size());
	assert(loopdepth<enum_block[splitting_depth].size());
	for(unsigned int n=0;n<=splitting_depth;n++) assert(enum_split[n]>0);
	assert(enum_block[splitting_depth][loopdepth]>0);
	assert_splits(splitting_depth);
#endif
	switch(mappedDataPart(ofwhat)) {
		case TRAIN: {tmp=splits[splitting_depth][active_class]._reduced_train->getNextBlock(loopdepth); break;}
		case TEST: {tmp=splits[splitting_depth][active_class]._reduced_test->getNextBlock(loopdepth); break;}
		case TRAINTEST: {
			assert(loopdepth<tt_phase[splitting_depth].size());
			assert(tt_phase[splitting_depth][loopdepth]==TRAIN || tt_phase[splitting_depth][loopdepth]==TEST);
			if(tt_phase[splitting_depth][loopdepth]==TRAIN) {
				tmp=splits[splitting_depth][active_class]._reduced_train->getNextBlock(loopdepth); 
				if(tmp==NULL) {
					tt_phase[splitting_depth][loopdepth]=TEST;
					tmp=splits[splitting_depth][active_class]._reduced_test->getFirstBlock(loopdepth);
				}
			} else {
				tmp=splits[splitting_depth][active_class]._reduced_test->getNextBlock(loopdepth);
			}
			if(tmp==NULL) tt_phase[splitting_depth][loopdepth]=NONE;
			break;}
		case ALL: {if(splitting_depth>0) tmp=splits[splitting_depth-1][active_class]._reduced_train->getNextBlock(loopdepth); else tmp=NULL; break;}
		case NONE: {tmp=NULL;}
	}
	if(tmp!=NULL) enum_block[splitting_depth][loopdepth]++; else {enum_block[splitting_depth][loopdepth]=0; return false;}
	return true;
}

// strong copy-constructor, hard-copies all structures
// be careful when used for cloning for concurrency
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::DataSplit::DataSplit(const typename Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::DataSplit& cs)
{
	notify("DataSplit strong copy constructor.");
	if(cs._train) _train.reset(new INTERVALCONTAINER(*cs._train)); //else _train.reset();
	if(cs._test) _test.reset(new INTERVALCONTAINER(*cs._test)); //else _test.reset();
	if(cs._reduced_train == cs._train) // && cs._reduced_test == cs._test
	{
		notify("DataSplit _reduced_* to be linked");
		_reduced_train=_train;
		_reduced_test=_test;
	} else {
		notify("DataSplit _reduced_* to be re-allocated");
		if(cs._reduced_train) _reduced_train.reset(new INTERVALCONTAINER(*cs._reduced_train)); //else _reduced_train.reset();
		if(cs._reduced_test) _reduced_test.reset(new INTERVALCONTAINER(*cs._reduced_test)); //else _reduced_test.reset();
	}
	_all.startidx=cs._all.startidx; _all.count=cs._all.count;
}

// weak copy-constructor, copies only references to PIntervaller structures
// be careful when used for cloning for concurrency
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::DataSplit::DataSplit(const typename Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::DataSplit& cs, int)
{
	notify("DataSplit weak copy constructor.");
	_train=cs._train; //if(_train) notify("TRAIN NOT EMPTY");
	_test=cs._test; //if(_test) notify("TEST NOT EMPTY");
	_reduced_train=cs._reduced_train; //if(_reduced_train) notify("RTRAIN NOT EMPTY");
	_reduced_test=cs._reduced_test; //if(_reduced_test) notify("RTEST NOT EMPTY");
	_all.startidx=cs._all.startidx; _all.count=cs._all.count;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
std::ostream& Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::print(std::ostream& os) const 
{
	if(dsp) {
		if(dsp->size()>0) {
			os << "Data_Accessor_Splitting() splitters: ";//<<std::endl; 
			typename std::vector<PSplitter>::const_iterator iter;
			for(iter=dsp->begin();iter!=dsp->end();iter++) {
				if(*iter) os << **iter << " "; else os << "nullptr "; //std::endl;
			}
		} else os << "Data_Accessor_Splitting()"; 
	}
	return os;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void Data_Accessor_Splitting<DATATYPE,IDXTYPE,INTERVALCONTAINER>::assert_splits(const int splitting_check) const
{
	// assumes Data_Accessor_Splitting() has been initialized; all data structures are checked to be valid and non-empty
#ifdef DEBUG
	assert(features>0); // No. of features
	assert(classes.size()>0); // No. of classes
	assert(dsp); //pridat kontrolu obsahu vektoru
	assert(dsp->size()>0); // No. of splitting levels
	// the following not to be checked here
	
	assert(splitting_check<(const int)dsp->size()); 
	// splitting_check == -1 by default
	// splitting_check > -1 checks that in each level 0..splitting_check the number of samples is set > 0

	// check correctness of data strucutre sizes
	assert(enum_split.size()==dsp->size());
	assert(splits.size()==dsp->size());
	assert(enum_block.size()==dsp->size());
	assert(tt_phase.size()==dsp->size());
	for(unsigned int n=0;n<dsp->size();n++) assert(splits[n].size()==classes.size());
	
	for(unsigned int n=0;n<dsp->size();n++)
	{
		for(unsigned int c=0;c<classes.size();c++) 
		{
			if((int)n<=splitting_check) assert((*dsp)[n]->get_n(c)>0);
			assert(splits[n][c]._train);
			assert(splits[n][c]._test);
			assert(splits[n][c]._reduced_train);
			assert(splits[n][c]._reduced_test);
			if(n==0) {
				assert(splits[n][c]._reduced_train==splits[n][c]._train);
				assert(splits[n][c]._reduced_test==splits[n][c]._test);
			} else {
				assert(splits[n][c]._reduced_train!=splits[n][c]._train);
				assert(splits[n][c]._reduced_test!=splits[n][c]._test);
			}
		}
	}
#endif
}


} // namespace
#endif // FSTDATAACCESSORSPLITTING_H ///:~
