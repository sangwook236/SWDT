#ifndef FSTSEARCHSEQ_H
#define FSTSEARCHSEQ_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    search_seq.hpp
   \brief   Defines interface for sequential search method implementations
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
#include <sstream>
#include "error.hpp"
#include "global.hpp"
#include "search.hpp"

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

enum SearchDirection {
  UNDEFINED=0,
  FORWARD=1,
  BACKWARD=2
};

/*! \brief Abstract class, defines interface for sequential search method implementations, provides common implementation of the "sequential step" operation

	\note To prevent misunderstanding: _generalization_level here does not relate to 
	classification peformance on unknown data, but to "generalization" of the course of search, 
	i.e., the number of features added/removed while generating candidate subsets in one sequential step
	(higher generalization=more candidates). See book Devijver, Kittler, 1982 for discussion.
*/
template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
class Search_Sequential : public Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{ // abstract class
public:
	typedef Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent;
	typedef boost::shared_ptr<EVALUATOR> PEvaluator;
	typedef boost::shared_ptr<CRITERION> PCriterion;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Search_Sequential(const PEvaluator evaluator):Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>(), _generalization_level(1), _initial_subset_enabled(false) {assert(evaluator); _evaluator=evaluator; notify("Search_Sequential constructor.");}
	virtual ~Search_Sequential() {notify("Search_Sequential destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) =0; //!< returns found subset + criterion value
	
	inline bool Step(const bool forward, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout); //!< modifies sub by adding/removing best/worst feature g-tuple, returns true if sub has been changed

	void set_generalization_level(const DIMTYPE g) {if(g>=1) _generalization_level=g;} //!< optional parameter to enlarge the number of considered candidates (g is the number of features added/removed in one Step)
	DIMTYPE get_generalization_level() const {return _generalization_level;}

	// to be used only if the search is intended to be started from a pre-specified initial subset, otherwise the standard FORWARD methods start from empty set and BACKWARD methods start from full set
	void enable_initial_subset() {_initial_subset_enabled=true;} 
	void disable_initial_subset() {_initial_subset_enabled=false;} 
	bool initial_subset_enabled() const {return _initial_subset_enabled;}
	
protected:
	PEvaluator _evaluator;
	DIMTYPE _generalization_level;
	bool _initial_subset_enabled; //!< by default all concrete search methods ignore the contents of supplied 'PSubset set' and initialize sub according to their definition; if enabled, then (if possible) the search() starts from supplied 'PSubset sub' contents
};


template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, class EVALUATOR>
bool Search_Sequential<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,EVALUATOR>::Step(const bool forward, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // modifies sub, returns true if sub changed
{
	assert(_evaluator);
	assert(sub);
	assert(crit);

	if(parent::output_detailed()) {std::ostringstream sos; sos << "Search_Sequential::Step(): sub->get_d()="<<sub->get_d()<<" forward="<<forward <<" sub=" << *sub<<endl; syncout::print(os,sos);}

	bool success=true;
	bool _mode_change=(sub->get_forward_mode()!=forward);

	if(_mode_change) sub->set_forward_mode(forward);
	if(!_evaluator->evaluate_candidates(result, sub, crit, _generalization_level,os)) success=false;
	if(_mode_change) sub->set_forward_mode(!forward);

	return success;
}


} // namespace
#endif // FSTSEARCHSEQ_H ///:~
