#ifndef FSTSEARCHBRANCHANDBOUNDIMPROVEDTHREADED_H
#define FSTSEARCHBRANCHANDBOUNDIMPROVEDTHREADED_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    search_branch_and_bound_improved_threaded.hpp
   \brief   Implements Threaded version of Improved Branch and Bound (IBB) method, i.e., with fully computed node ordering
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
  * FST3 software is availables free of charge for non-commercial use. 
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
#include <cstdlib> //rand
#include "search_branch_and_bound_improved.hpp"
#include "error.hpp"
#include "global.hpp"
#include "result_tracker.hpp"
#include "candidate_evaluator_threaded.hpp"

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

/*! \brief Implements threaded Improved Branch and Bound (IBB) method, i.e., with fully computed node ordering

	\warning All Branch & Bound feature selection algorithms require the used
	CRITERION to be monotonic with respect to cardinality. More precisely, it must
	hold that removing a feature from a set MUST NOT increase criterion value.
	Otherwise there is no guarantee as of the optimality of obtained results
	with respect to the used criterion.

	\note All Branch & Bound algorithms by definition yield the solution with the 
	same maximum criterion value, therefore the main concern regarding particular
	Branch & Bound algorithm is only its search speed.

	\note Threading is implemented here wherever it was straightforward to do so.
	At various stages the algorithm thus switches beween sequential and threaded
	evaluation (e.g., node ordering is parallelized, evaluation of leafs at the end
	of paths is not). Taking better use of concurrency would require much more effort
	or conceptual changes in the algorithm itself. See specialized research resources for 
	more information on this problem.
	
	\warning This implementation unfortunately is not capable of accelerating feature selection
	process if any of the monotonous criteria implemented currently in the FST library
	are used; this is because the evaluation speed of any of the implemented monotonous criteria 
	is too high when compared to threading overhead. Unless you need to run IBB with
	a computationally expensive monotonous criterion (if such is custom implemented),
	we strongly recommend to stay with the standard non-concurrent version of IBB.

	\note Due to possibly high number of subsets to be tested expect
	excessive computational time. 
	
	\note Result tracking in case of Branch & Bound algorithms records only results
	of target cardinality.
*/

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads=2>
class Search_Branch_And_Bound_Improved_Threaded : public Search_Branch_And_Bound_Improved<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> grandgrandparent;
	typedef Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> grandparent;
	typedef typename grandparent::PCriterion PCriterion;
	typedef typename grandparent::PSubset PSubset;
	typedef typename grandparent::Node Node;
	typedef typename grandparent::NodeType NodeType;
	Search_Branch_And_Bound_Improved_Threaded():Search_Branch_And_Bound_Improved<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>() 
		{BOOST_STATIC_ASSERT(max_threads>0); notify("Search_Branch_And_Bound_Improved_Threaded constructor.");}
	virtual ~Search_Branch_And_Bound_Improved_Threaded() {notify("Search_Branch_And_Bound_Improved_Threaded destructor.");}

	// for search() implementation see grandparent class

	virtual std::ostream& print(std::ostream& os) const {os << "Improved Branch & Bound (IBB)  [Search_Branch_And_Bound_Improved_Threaded(threads="<<max_threads<<")]"; return os;};
	
protected:
	Candidate_Evaluator_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads> _evaluator;
	// the following may be overriden in descendant Branch and Bound specialization classes
	virtual void process_leafs(); //!< overridden to implement threading
	virtual void initialize(const DIMTYPE d, const DIMTYPE n, const PCriterion crit); //!< called before search - enables set-up of additional structures in descendants
	virtual void pre_evaluate_availables(); //!< assign values to each feature in availables - to be used for node ordering
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
void Search_Branch_And_Bound_Improved_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::initialize(const DIMTYPE d, const DIMTYPE n, const PCriterion crit)
{ // called before search - enables set-up of additional structures in descendants
	_evaluator.initialize(crit, n);
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
void Search_Branch_And_Bound_Improved_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::process_leafs()
{
	assert(currentset);
	assert(grandparent::get_criterion());
	const PSubset &currentset=grandparent::get_currentset();
	typename grandparent::PNode avail;
	for(bool got=getFirstAvailable(avail);got;got=getNextAvailable(avail)) 
	{
		assert(currentset->selected_raw(avail->feature));
		currentset->deselect_raw(avail->feature);
		_evaluator.add_candidate(currentset,avail->feature);
		currentset->select_raw(avail->feature);
	}
	RETURNTYPE value;
	DIMTYPE feature;
	PSubset sub;
	for(bool got=_evaluator.getFirstResult(value,sub,feature);got;got=_evaluator.getNextResult(value,sub,feature)) 
	{
		grandparent::update_bound(value,sub);  //adds to tracker!
	}
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION, unsigned int max_threads>
void Search_Branch_And_Bound_Improved_Threaded<RETURNTYPE,DIMTYPE,SUBSET,CRITERION,max_threads>::pre_evaluate_availables() 
{
	assert(currentset);
	assert(grandparent::get_criterion());
	const PSubset &currentset=grandparent::get_currentset();
	// pre-evaluate available features before next tree level construction
	RETURNTYPE value;
	DIMTYPE feature;
	typename grandparent::PNode avail;
	for(bool got=getFirstAvailable(avail);got;got=getNextAvailable(avail)) 
	{
		assert(currentset->selected_raw(avail->feature));
		currentset->deselect_raw(avail->feature);
		_evaluator.add_candidate(currentset,avail->feature);
		currentset->select_raw(avail->feature);
	}
	PSubset dummysub;
	for(bool got=_evaluator.getFirstResult(value,dummysub,feature);got;got=_evaluator.getNextResult(value,dummysub,feature)) 
	{
		if(getAvailable(feature,avail))
		{
			avail->value=value;
			avail->type=grandparent::COMPUTED;
		}		
	}
}

} // namespace
#endif // FSTSEARCHBRANCHANDBOUNDIMPROVEDTHREADED_H ///:~
