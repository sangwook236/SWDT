#ifndef FSTSEARCHBRANCHANDBOUND_H
#define FSTSEARCHBRANCHANDBOUND_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    search_branch_and_bound.hpp
   \brief   Implements Branch and Bound template method as basis for more advanced B&B implementations
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
  * FST3 software is _available free of charge for non-commercial use. 
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
#include <ctime>
#include <cstdlib> //rand
#include <vector>
#include <list>
#include "error.hpp"
#include "global.hpp"
#include "stopwatch.hpp"
#include "search.hpp"
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

/*! \brief Implements Branch and Bound template method as basis for more advanced B&B implementations

	This is the abstract basis of concrete Branch & Bound implementations.

	\warning All Branch & Bound feature selection algorithms require the used
	CRITERION to be monotonic with respect to cardinality. More precisely, it must
	hold that removing a feature from a set MUST NOT increase criterion value.
	Otherwise there is no guarantee as of the optimality of obtained results
	with respect to the used criterion.

	\note Due to possibly high number of subsets to be tested expect
	excessive computational time. 
	
	\note Result tracking in case of Branch & Bound algorithms records only results
	of target cardinality.
*/

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
class Search_Branch_And_Bound : public Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>{
public:
	typedef Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION> parent;
	typedef boost::shared_ptr<CRITERION> PCriterion;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Search_Branch_And_Bound():Search<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>() {notify("Search_Branch_And_Bound constructor.");}
	virtual ~Search_Branch_And_Bound() {notify("Search_Branch_And_Bound destructor.");}

	// NOTE: crit is expected to be already initialize()d before use here
	virtual bool search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os=std::cout); //!< returns found subset of target_d features (optimizes cardinality if target_d==0) + _criterion value

	virtual std::ostream& print(std::ostream& os) const {os << "Search_Branch_And_Bound() abstract class"; return os;};

protected:
	typedef enum {UNKNOWN, RANDOM, PREDICTED, COMPUTED} NodeType;
	//! Structure representing search tree node
	typedef struct {
		DIMTYPE    feature; 
		RETURNTYPE value;
		NodeType   type;
		bool       valid; // for use in _available
	} Node;
	typedef Node* PNode;
	typedef list<Node> ListNode;
	typedef vector<ListNode> VectorListNode;
	typedef typename ListNode::iterator ListNodeIter;
	typedef typename vector<ListNode>::iterator TreeLevelIter;
	typedef vector<Node> VectorNode;

private:
	// the following structures need to be valid throughout the course of one search
	PCriterion _criterion;
	
	VectorListNode _tree;
	
	ListNodeIter   _nodeiter; //!< used in get*Node()
	DIMTYPE        _availiter;

	VectorNode     _available; //!< Node::feature takes the role of 1-selected/0-unselected indicator
	DIMTYPE         get_available_size() {DIMTYPE c=0; for(DIMTYPE i=0;i<_available.size();i++) if(_available[i].valid) c++; return c;}
	void            available_add(const DIMTYPE feature) {assert(feature>=0 && _available.size()>feature && !_available[feature].valid); _available[feature].valid=true; _available[feature].type=UNKNOWN;}
	void            available_remove(const DIMTYPE feature) {assert(feature>=0 && _available.size()>feature && _available[feature].valid); _available[feature].valid=false;}

	PSubset        _currentset; //!< _currentset subset to be evaluated
	void            currentset_add(const DIMTYPE feature) {assert(!_currentset->selected_raw(feature)); _currentset->select_raw(feature);}
	void            currentset_remove(const DIMTYPE feature) {assert(_currentset->selected_raw(feature)); _currentset->deselect_raw(feature);}

	PSubset        _tempset; //!< temporary copy of _currentset subset to be modified and evaluated
	
	PSubset        _boundset;
	RETURNTYPE     _boundvalue;
	bool           _boundvalid;
	
	DIMTYPE         k; //!< current tree level
	DIMTYPE         n; //!< no. of all features
	DIMTYPE         d; //!< no. of features to be selected
	
protected:
	PCriterion const &get_criterion() const {assert(_criterion); return _criterion;}

	RETURNTYPE get_bound_value() const {return _boundvalue;}
	bool       is_bound_valid() const {return _boundvalid;}
	
	int     get_k() const {return k;}
	DIMTYPE get_n() const {return n;}
	DIMTYPE get_d() const {return d;}
	
	bool getFirstNode(PNode &nod) {assert(k>=0 && k<n-d+1 && _tree.size()==n-d); if(_tree[k].empty()) return false; else {_nodeiter=_tree[k].begin(); nod=&(*_nodeiter); return true;}}
	bool getNextNode(PNode &nod) {assert(k>=0 && k<n-d+1 && _tree.size()==n-d); if(++_nodeiter==_tree[k].end()) return false; else {nod=&(*_nodeiter); return true;}}

	bool getFirstAvailable(PNode &avail) {assert(_available.size()>0); for(_availiter=0; _availiter<_available.size() && !_available[_availiter].valid; _availiter++); if(_availiter<_available.size()) {avail=&(_available[_availiter]); return true;} else return false;}
	bool getNextAvailable(PNode &avail) {assert(_available.size()>0 && _availiter<_available.size()); _availiter++; while(_availiter<_available.size() && !_available[_availiter].valid) {_availiter++;} if(_availiter<_available.size()) {avail=&(_available[_availiter]); return true;} else return false;}
	bool getAvailable(DIMTYPE idx, PNode &avail) {assert(idx>=0 && idx<_available.size()); if(idx<_available.size()) {avail=&(_available[idx]); return true;} else return false;}

protected:	
	PSubset& get_currentset(){return _currentset;}
	Node&    get_parent_node(){assert(k>0 && _tree.size()>k && !_tree[k-1].empty()); return _tree[k-1].back();}
	Node&    get_current_node(){assert(_tree.size()>k && !_tree[k].empty()); return _tree[k].back();}
	
	std::ostream* shared_os; //!< to enable sharing the same stream across the methods below
	
	void setup_default_structures(const DIMTYPE d, const DIMTYPE n); //!< initializes tree memory
	void update_bound(const RETURNTYPE val, const PSubset &sub);
	void make_tree_level();
	void process_rightmost_path();

	boost::shared_ptr<StopWatch> swatch;
protected:
	// the following may/must be overriden in descendant Branch and Bound specialization classes (to implement specific functionality or threading)
	virtual void process_leafs(); //!< can be overridden to implement prediction information learning, threading etc.
	virtual void initialize(const DIMTYPE d, const DIMTYPE n, const PCriterion crit)=0; //!< called before search - enables set-up of additional structures in descendants
	virtual void pre_evaluate_availables()=0; //!< assign values to each feature in _available - to be used for node ordering
	virtual void post_process_tree_level()=0; //!< enables to substitute missing COMPUTED values in nodes just after level creation, if needed
	virtual bool cut_possible()=0; //!< tests _currentset node for the possibility to cut its sub-branch
};

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::setup_default_structures(const DIMTYPE d, const DIMTYPE n)
{
	assert(d>0);
	assert(d<n);
	_tree.resize(n-d); // logically n-d+1, but last (leaf) level does is not stored but evaluated directly
	for(TreeLevelIter i=_tree.begin();i!=_tree.end();i++) i->clear();
	_available.resize(n); for(DIMTYPE f=0;f<n;f++) {_available[f].feature=f; _available[f].valid=true;}
	_currentset.reset(new SUBSET(n)); _currentset->select_all_raw();
	_tempset.reset(new SUBSET(n));
	_boundset.reset(new SUBSET(n));	_boundvalid=false;
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::update_bound(const RETURNTYPE val, const PSubset &sub)
{
	assert(sub);
	assert(_boundset);
	assert(sub->get_n()==_boundset->get_n());
	assert(swatch);
	if(!_boundvalid || val>_boundvalue) {
		_boundvalue=val;
		_boundset->stateless_copy(*sub);
		_boundvalid=true;
		if(parent::output_normal()) {std::ostringstream sos; sos << "NEW BOUND (val="<<val<<"): " << *sub << std::endl << *swatch << std::endl << std::endl << std::flush; syncout::print(*shared_os,sos);}
	}
	if(parent::result_tracker_active()) parent::_tracker->add(val,sub);
}	

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::make_tree_level() // make k-th tree level
{
	// based on k and no. of _available features
	assert(k>0);
	assert(k<n-d+1);
	assert(_tree.size()==n-d);
	assert(_tree[k].empty());
	assert(_available.size()==n);
	
	pre_evaluate_availables(); // assign values to candidates to enable sorting into the tree level to be constructed here
	
	const DIMTYPE levelsize=get_available_size()-(n-d-k+1);
	assert(levelsize>0);
	assert(levelsize<=get_available_size());

	// sort-in levelsize features according to value (in case of type UNKNOWN feature put at the end of list)
	PNode avail;
	for(bool got=getFirstAvailable(avail);got;got=getNextAvailable(avail))
	{
		DIMTYPE pos=0;
		ListNodeIter it=_tree[k].begin();
		while(it!=_tree[k].end() && it->value<avail->value) {++it; ++pos;}
		if(pos<levelsize) {
			_tree[k].insert(it,*avail);
			if(_tree[k].size()>levelsize) _tree[k].pop_back();
		}
	}
	assert(_tree[k].size()==levelsize);
	for(ListNodeIter it=_tree[k].begin();it!=_tree[k].end();it++) {
		available_remove(it->feature);
	}
	
	post_process_tree_level(); // enable to re-evaluate chosen candidate values (e.g., supply missing true _criterion values, etc.)
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::process_rightmost_path()
{
	assert(_tempset);
	assert(_currentset);
	assert(_available.size()==n);
	assert(get_available_size()>0);
	assert(_criterion);
	_tempset->stateless_copy(*_currentset);
	
	RETURNTYPE val;
	PNode avail;
	
	for(bool got=getFirstAvailable(avail);got;got=getNextAvailable(avail)) _tempset->deselect_raw(avail->feature);
	if(!_criterion->evaluate(val,_tempset)) throw FST::fst_error("Criterion evaluation failure."); 
	update_bound(val,_tempset); //adds to tracker
	if(parent::output_detailed()) {std::ostringstream sos; sos << "rightmost leaf (val="<<val<<"): " << *_tempset << std::endl << std::endl << std::flush; syncout::print(*shared_os,sos);}
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
void Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::process_leafs()
{
	assert(_currentset);
	assert(get_criterion());
	RETURNTYPE value;
	PNode avail;
	for(bool got=getFirstAvailable(avail);got;got=getNextAvailable(avail)) 
	{
		currentset_remove(avail->feature);
		if(!get_criterion()->evaluate(value,_currentset)) throw FST::fst_error("Criterion evaluation failure."); 
		update_bound(value,_currentset); //adds to tracker
		if(parent::output_detailed()) {std::ostringstream sos; sos << "leaf (val="<<value<<"): " << *_currentset << std::endl << std::endl << std::flush; syncout::print(*shared_os,sos);}
		currentset_add(avail->feature);
	}
}

template<class RETURNTYPE, typename DIMTYPE, class SUBSET, class CRITERION>
bool Search_Branch_And_Bound<RETURNTYPE,DIMTYPE,SUBSET,CRITERION>::search(const DIMTYPE target_d, RETURNTYPE &result, const PSubset sub, const PCriterion crit, std::ostream& os) // returns found subset + _criterion value
{
	swatch.reset(new StopWatch);
	notify("Search_Branch_And_Bound::search().");
	assert(sub);
	assert(crit);
	n=sub->get_n();
	assert(target_d>0 && target_d<n);
	d=target_d;
	assert(d>0 && d<=n);
	shared_os=&os;

	if(parent::result_tracker_active()) parent::_tracker->set_output_detail(parent::get_output_detail());
		
	if(parent::output_normal()) {
		std::ostringstream sos; 
		sos << "---------------------------------------" << std::endl;
		sos << "Starting " << *this << std::endl;
		sos << "with Criterion: " << *crit << std::endl;
		sos << "Target subset size set to: " << target_d << std::endl;
		sos << std::endl << std::flush;
		syncout::print(os,sos);
	}

	if(n==d) {
		sub->select_all_raw();
		if(!crit->evaluate(result,sub)) throw FST::fst_error("Criterion evaluation failure."); 
		return true;
	}
	
	setup_default_structures(d,n);
	
	// initialize root
	Node root; // Node::feature and Node::valid are not used in root
	_tempset->select_all_raw(); 
	if(!crit->evaluate(root.value,_tempset)) throw FST::fst_error("Criterion evaluation failure."); 
	root.type=COMPUTED;
	_tree[0].push_back(root);
		
	initialize(d,n,crit); // to enable overloaded functionality
	_criterion=crit;
		
	DIMTYPE lastcut_k=n-d; // helpers to prevent excessive dumps

	DIMTYPE feature;
	k=1; // start with the first level below root
	do{
		// traverse tree branch down
		while(k>0 && k<n-d) {

			make_tree_level(); // make k-th level, choose |_available|-(n-d-k+1) nodes from _available and use them to make next level
			process_rightmost_path();
			
			// recursively cut all sub-branches that can be cut
			while(k>0 && cut_possible())
			{
				assert(!_tree[k].empty());
				if(k<=lastcut_k) { // with time this condition should prevent excessive dumps
					if(parent::output_normal()) {std::ostringstream sos; sos << "CUT (k="<<k<<", val="<<_tree[k].back().value<<", bound="<<get_bound_value()<<", feature=" << _tree[k].back().feature<<")  " << std::endl << std::endl; syncout::print(os,sos);}
					lastcut_k=k;
				}
				feature=_tree[k].back().feature; assert(feature>=0 && feature<n);
				available_add(feature);
				_tree[k].pop_back();
				while(k>0 && _tree[k].empty()) {
					--k;
					if(k>0) {
						assert(!_tree[k].empty());
						feature=_tree[k].back().feature; assert(feature>=0 && feature<n);
						currentset_add(feature);
						available_add(feature);
						_tree[k].pop_back();
					}
				} 
			}
			// no more cutting possible, proceed down to next level (or finish if k==0)
			if(k>0) {
				feature=_tree[k].back().feature; assert(feature>=0 && feature<n);
				currentset_remove(feature);
				++k;
			}
		}
		// evaluate all leafs that are descendants of the just visited node
		if(k==n-d) {
			process_leafs(); //depends on _currentset, _available, not on k
			// backtrack up to closest branching ascendant node
			do {
				--k;
				if(k>0) {
					assert(!_tree[k].empty());
					feature=_tree[k].back().feature; assert(feature>=0 && feature<n);
					currentset_add(feature);
					available_add(feature);
					_tree[k].pop_back();
				}
			} while(k>0 && _tree[k].empty());
		}
		// proceed down through the next rightmost subbranch to the left from the last one
		if(k>0) {
			assert(!_tree[k].empty());
			feature=_tree[k].back().feature; assert(feature>=0 && feature<n);
			currentset_remove(feature);
			++k;
		}
		
	} while(k>0);

	assert(_boundset);
	assert(sub->get_n()==_boundset->get_n());
	sub->stateless_copy(*_boundset);
	result=_boundvalue;
	if(parent::output_normal()) {
		std::ostringstream sos; 
		sos << "......................................." << std::endl;
		sos << "Branch and Bound search finished. " << *swatch << std::endl;
		sos << "Search result: "<< std::endl << *sub << std::endl << "Criterion value: " << result << std::endl << std::endl << std::flush;
		syncout::print(os,sos);
	}
	swatch.reset();
	return true;
}

} // namespace
#endif // FSTSEARCHBRANCHANDBOUND_H ///:~
