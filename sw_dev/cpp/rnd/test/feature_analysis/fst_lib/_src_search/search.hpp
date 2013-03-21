#ifndef FSTSEARCH_H
#define FSTSEARCH_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    search.hpp
   \brief   Defines interface for search method implementations
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
#include <ctime>
#include "error.hpp"
#include "global.hpp"
#include "result_tracker.hpp"

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

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search;

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
std::ostream& operator<<(std::ostream& os, const Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>& sr);

//! abstract class, defines interface for search method implementations
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search { // abstract class
	friend std::ostream& operator<< <>(std::ostream& os, const Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>& sr);
public:
	typedef boost::shared_ptr<CRITERION> PCriterion;
	typedef boost::shared_ptr<SUBSET> PSubset;
	typedef boost::shared_ptr<Result_Tracker<RETURNTYPE,SUBSET> > PResultTracker;
	Search(): _detail(NORMAL) {}
	virtual ~Search() {}

	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) =0; //!< returns found subset of target_d features (optimizes if target_d==0)+ criterion value

	void enable_result_tracking(PResultTracker tracker) {_tracker=tracker;} //!< assigns a result tracker to the search process, i.e., enables logging of all evaluated subsets for future reuse (re-evaluation, etc.)
	void disable_result_tracking() {_tracker.reset();}
	bool result_tracker_active() const {return _tracker;}

	void set_output_detail(OutputDetail detail) {_detail=detail;} //!< sets the amount of information that is logged to output stream throughout the course of search
	OutputDetail get_output_detail() const {return _detail;}
	bool output_normal() const {return _detail==NORMAL || _detail==DETAILED;}
	bool output_detailed() const {return _detail==DETAILED;}
	
	virtual std::ostream& print(std::ostream& os) const {return os;}

protected:
	PResultTracker _tracker;
	OutputDetail _detail;
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
std::ostream& operator<<(std::ostream& os, const Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>& sr) {
	return sr.print(os);
}

} // namespace
#endif // FSTSEARCH_H ///:~
