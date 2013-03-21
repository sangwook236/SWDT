#ifndef FSTRESULTTRACKERFEATURESTATS_H
#define FSTRESULTTRACKERFEATURESTATS_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    result_tracker_feature_stats.hpp
   \brief   Computes feature occurence statistics in a series of evaluated subsets
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
#include <list>
#include "error.hpp"
#include "global.hpp"
#include "result_tracker_dupless.hpp"

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

/*! \brief Collects evaluated subsets to eventually provide Dependency-Aware Feature Ranking coefficients.

	Dependency-Aware Feature ranking (DAF) is a new type of ranking method especially
	suitable for very-high-dimensional feature selection. Unlike standard individual
	feature ranking, the DAF ranking reflects "average feature quality in context"
	and as such is capable of providing significantly better results than BIF
	(provided the data actually do contain mutually dependent features). The method
	has been described in combination with Monte Carlo based feature selection
	permitting Wrapper Criteria even in very-high-dimensional setting.
	For usage see \ref example35 and \ref example36. For more detailed
	information see UTIA Technical Report No. 2295.
*/ 
template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
class Result_Tracker_Feature_Stats : public Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> {
public:
	typedef Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> parent;
	typedef typename parent::ResultRec ResultRec;
	typedef ResultRec* PResultRec;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Result_Tracker_Feature_Stats(const IDXTYPE capacity_limit=0):Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>(capacity_limit), _n(0) {_itersize[0]=0; _itersize[1]=0; _itersize[2]=0; notify("Result_Tracker_Feature_Stats constructor.");}
	Result_Tracker_Feature_Stats(const Result_Tracker_Feature_Stats& rtfs):Result_Tracker_Dupless<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET>(rtfs), _n(0) {_itersize[0]=0; _itersize[1]=0; _itersize[2]=0; notify("Result_Tracker_Feature_Stats copy-constructor.");}
	virtual ~Result_Tracker_Feature_Stats() {notify("Result_Tracker_Feature_Stats destructor.");}
	
	bool compute_stats(std::ostream& os=std::cout);
	bool print_stats(std::ostream& os=std::cout); 

	bool getFirstDAF(RETURNTYPE &value, DIMTYPE &feature, const unsigned int DAFidx=0) const;
	bool getNextDAF(RETURNTYPE &value, DIMTYPE &feature, const unsigned int DAFidx=0) const;

	Result_Tracker_Feature_Stats* clone() const {throw fst_error("Result_Tracker_Feature_Stats::clone() not supported, use Result_Tracker_Feature_Stats::stateless_clone() instead.");}
	Result_Tracker_Feature_Stats* sharing_clone() const {throw fst_error("Result_Tracker_Feature_Stats::sharing_clone() not supported, use Result_Tracker_Feature_Stats::stateless_clone() instead.");}
	Result_Tracker_Feature_Stats* stateless_clone() const;

	virtual std::ostream& print(std::ostream& os) const {os << "Result_Tracker_Feature_Stats(limit=" << parent::_capacity_limit << ") size " << parent::results.size(); return os;}

protected:
	DIMTYPE _n;
	//! Structure to gather feature occurence statistics over probe subset evaluations
	typedef struct {
		RETURNTYPE freq_is, freq_isnot; // no of subsets containing/not-containing this feature
		RETURNTYPE mean_is, mean_isnot; // primary criterion mean value over subsets containing/not-containing this feature
		RETURNTYPE stdev_is, stdev_isnot; // primary criterion stddev over subsets containing/not-containing this feature
		// DAF0 indicator=  mean_is-mean_isnot
		// DAF1 indicator= (mean_is-mean_isnot)/((freq_is/freq)*stdev_is+(freq_isnot/freq)*stdev_isnot)
		// for DAF3 see UTIA Tech Report No. 2295
		RETURNTYPE daf[3]; 
		bool daf_valid[3]; 
	} FeatureStat;
	vector<FeatureStat> _stats; // for each feature over all subset sizes + then for each feature and each subset size

	//! Structure to gather probe subset cardinality statistics
	typedef struct {
		RETURNTYPE freq; // no of subsets
		RETURNTYPE mean;
		RETURNTYPE stdev;
		bool valid;
	} SubSizeStat;
	vector<SubSizeStat> _dstat; // for each feature over all subset sizes
	
	typedef vector<DIMTYPE> ORDERTYPE;
	ORDERTYPE _order[3];  // to keep feature ordering according to DAF0, DAF1 and DAF2
	mutable DIMTYPE _itersize[3]; // iterator over feature ordering according to DAF0, DAF1 and DAF2
};

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>* Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::stateless_clone() const
{
	Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET> *clone=new Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>(*this);
	clone->set_cloned();
	return clone;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::compute_stats(std::ostream& os)
{
	if(parent::results.empty()) return false;
	
	typename parent::CONSTRESULTSITER iter=parent::results.begin();
	// get n from first recorded sub (assumes all others have the same n)
	_n=iter->sub->get_n();
	_stats.resize(_n);
	for(DIMTYPE f=0;f<_n;f++) {
		_stats[f].freq_is=_stats[f].freq_isnot=_stats[f].mean_is=_stats[f].mean_isnot=_stats[f].stdev_is=_stats[f].stdev_isnot=0;
		_stats[f].daf_valid[0]=false;
		_stats[f].daf_valid[1]=false;
		_stats[f].daf_valid[2]=false;
	}
	_dstat.resize(_n+1);
	for(DIMTYPE ss=1;ss<=_n;ss++) {
		_dstat[ss].mean=_dstat[ss].stdev=_dstat[ss].freq=0;
		_dstat[ss].valid=false;
	}

	if(parent::output_normal()) {std::ostringstream sos; sos << std::endl << "compute_stats() 1/4.." << std::endl << std::flush; syncout::print(os,sos);}
	// gather initial stats
	for(iter=parent::results.begin();iter!=parent::results.end();iter++)
	{
		assert(iter->sub->get_n()==_n);
		DIMTYPE d=iter->sub->get_d();
		assert(d>0 && d<=_n);
		for(DIMTYPE f=0;f<_n;f++) {
			if(iter->sub->selected_raw(f)) {
				_stats[f].freq_is++;
				_stats[f].mean_is+=iter->value;
			} else {
				_stats[f].freq_isnot++;
				_stats[f].mean_isnot+=iter->value;
			}
		}
		_dstat[d].mean+=iter->value;
		_dstat[d].freq++;
	}
	
	// calculate mean values
	for(DIMTYPE f=0;f<_n;f++)
	{
		if(_stats[f].freq_is>0) _stats[f].mean_is/=_stats[f].freq_is;
		if(_stats[f].freq_isnot>0) _stats[f].mean_isnot/=_stats[f].freq_isnot;
	}
	for(DIMTYPE ss=1;ss<=_n;ss++) if(_dstat[ss].freq>0) {
		_dstat[ss].mean/=_dstat[ss].freq;
	}
	if(parent::output_normal()) {std::ostringstream sos; sos << "compute_stats() 2/4.." << std::endl << std::flush; syncout::print(os,sos);}
		
	// gather more stats
	for(iter=parent::results.begin();iter!=parent::results.end();iter++)
	{
		assert(iter->sub->get_n()==_n);
		DIMTYPE d=iter->sub->get_d();
		assert(d>0 && d<=_n);
		for(DIMTYPE f=0;f<_n;f++) {
			if(iter->sub->selected_raw(f)) {
				_stats[f].stdev_is+=(_stats[f].mean_is-iter->value)*(_stats[f].mean_is-iter->value);
			} else {
				_stats[f].stdev_isnot+=(_stats[f].mean_isnot-iter->value)*(_stats[f].mean_isnot-iter->value);
			}
		}
		_dstat[d].stdev+=(_dstat[d].mean-iter->value)*(_dstat[d].mean-iter->value);
	}
	
	// calculate stddev values
	for(DIMTYPE f=0;f<_n;f++)
	{
		if(_stats[f].freq_is>0) _stats[f].stdev_is=sqrt(_stats[f].stdev_is/_stats[f].freq_is);
		if(_stats[f].freq_is>0) _stats[f].stdev_isnot=sqrt(_stats[f].stdev_isnot/_stats[f].freq_isnot);
	}
	for(DIMTYPE ss=1;ss<=_n;ss++) if(_dstat[ss].freq>0) {
		_dstat[ss].stdev=sqrt(_dstat[ss].stdev/_dstat[ss].freq);
		_dstat[ss].valid=true;
	}
	
	
	if(parent::output_normal()) {std::ostringstream sos; sos << "compute_stats() 3/4.." << std::endl << std::flush; syncout::print(os,sos);}
	// calculate DAF ranks
	for(DIMTYPE f=0;f<_n;f++)
	{
		if(_stats[f].freq_is>0 && _stats[f].freq_isnot>0 && (_stats[f].stdev_is>0 || _stats[f].stdev_isnot>0) ) 
		{
			RETURNTYPE freq=_stats[f].freq_is+_stats[f].freq_isnot;
			//DAF1
			_stats[f].daf[1]=(_stats[f].mean_is-_stats[f].mean_isnot)/( (_stats[f].freq_is/freq)*_stats[f].stdev_is + (_stats[f].freq_isnot/freq)*_stats[f].stdev_isnot );
			_stats[f].daf_valid[1]=true;
			//DAF0
			_stats[f].daf[0]=_stats[f].mean_is-_stats[f].mean_isnot;
			_stats[f].daf_valid[0]=true;
		} 
		else _stats[f].daf_valid[1]=_stats[f].daf_valid[0]=false;
	}
	//DAF2
	vector<RETURNTYPE> mi1,mi2; mi1.resize(_n+1); mi2.resize(_n+1);
	vector<IDXTYPE> cnt1,cnt2; cnt1.resize(_n+1); cnt2.resize(_n+1);
	for(DIMTYPE f=0;f<_n;f++) {mi1[f]=mi2[f]=0; cnt1[f]=cnt2[f]=0;}
	for(iter=parent::results.begin();iter!=parent::results.end();iter++)
	{
		assert(iter->sub->get_n()==_n);
		DIMTYPE d=iter->sub->get_d();
		assert(d>0 && d<=_n);
		
		if(_dstat[d].stdev>0) {
			for(DIMTYPE f=0;f<_n;f++) {
				if(iter->sub->selected_raw(f)) {mi1[f]+=iter->value/_dstat[d].stdev; cnt1[f]++;}
				else {mi2[f]+=iter->value/_dstat[d].stdev; cnt2[f]++;}
			}
		}
	}	
	for(DIMTYPE f=0;f<_n;f++) {
		if(cnt1[f]==0 || cnt2[f]==0) _stats[f].daf[2]=0; else
		{
			mi1[f]/=cnt1[f];
			mi2[f]/=cnt2[f];
			_stats[f].daf[2]=mi1[f]-mi2[f];
		}
		_stats[f].daf_valid[2]=true;
	}

	if(parent::output_normal()) {std::ostringstream sos; sos << "compute_stats() 4/4.." << std::endl << std::endl << std::flush; syncout::print(os,sos);}
	
	// order features according to DAF0, DAF1 and DAF2
	for(unsigned int DAFidx=0; DAFidx<3; DAFidx++)
	{
		_order[DAFidx].clear();
		for(DIMTYPE f=0;f<_n;f++)
		{
			if(_stats[f].daf_valid[DAFidx]) {
				typename ORDERTYPE::iterator iter=_order[DAFidx].begin();
				while(iter!=_order[DAFidx].end() && _stats[f].daf[DAFidx]<_stats[*iter].daf[DAFidx]) iter++;
				_order[DAFidx].insert(iter,f);
			}
		}
		if(_order[DAFidx].size()<_n && parent::output_normal()) {std::ostringstream sos; sos << "WARNING: DAF"<<DAFidx<<" rank could not be computed for all features due to insufficient number of evaluated probes subsets." << std::endl << std::flush; syncout::print(os,sos);}
	}
	if(parent::output_normal()) {
		RETURNTYPE minfreq=_stats[0].freq_is; if(_stats[0].freq_isnot<_stats[0].freq_is) minfreq=_stats[0].freq_isnot;
		for(DIMTYPE f=1;f<_n;f++) {
			if(_stats[f].freq_isnot<minfreq) minfreq=_stats[f].freq_isnot;
			if(_stats[f].freq_is<minfreq) minfreq=_stats[f].freq_is;
		}
		{
			std::ostringstream sos; 
			sos << "Lowest feature evaluation frequency over all probes is "<<minfreq<<"." << std::endl;
			if(minfreq<100) sos << "WARNING: To ensure reliable DAF estimates the minimum feature evaluation frequency should be at least in the order of hundreds. More probes may be needed." << std::endl;
			sos << std::endl << std::flush;
			syncout::print(os,sos);
		}
	}

	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::print_stats(std::ostream& os)
{
	std::ostringstream sos;
	if(_n<1) {sos << "Result_Tracker_Feature_Stats:: Nothing to print out." << std::endl; return false;}
	{
		sos << "Probe subset stats for each (valid) cardinality:"<< std::endl;
		for(DIMTYPE ss=1;ss<=_n;ss++) 
			if(_dstat[ss].valid) sos << " d: "<< ss << "  mean=" << _dstat[ss].mean << ", stddev=" << _dstat[ss].stdev << ", freq=" << _dstat[ss].freq << std::endl;
		sos << std::endl << "Feature stats over all probe subsets:"<< std::endl;
		for(DIMTYPE f=0;f<_n;f++) {
			sos << " F:"<<f<< " daf0 = "; if(_stats[f].daf_valid[0]) sos << _stats[f].daf[0]; else sos << "n/a"; 
			sos << ", daf1 = "; if(_stats[f].daf_valid[1]) sos << _stats[f].daf[1]; else sos << "n/a";
			sos << ", daf2 = "; if(_stats[f].daf_valid[2]) sos << _stats[f].daf[2]; else sos << "n/a";
			sos<<",  freq "<<_stats[f].freq_is<<"|"<<_stats[f].freq_isnot<<",  mean ";
			if(_stats[f].freq_is>0) sos<<_stats[f].mean_is; else sos<< "n/a";
			sos <<"|";
			if(_stats[f].freq_isnot>0) sos<<_stats[f].mean_isnot; else sos<< "n/a";
			sos<<", stdev ";
			if(_stats[f].freq_is>0) sos<<_stats[f].stdev_is; else sos<<"n/a";
			sos<<"|";
			if(_stats[f].freq_isnot>0) sos<<_stats[f].stdev_isnot; else sos << "n/a";
			sos<<std::endl;
		}
		sos<<std::endl;
	}
	syncout::print(os,sos);
	return true;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::getFirstDAF(RETURNTYPE &value, DIMTYPE &feature, const unsigned int DAFidx) const
{
	assert(0<=DAFidx && DAFidx<3);
	_itersize[DAFidx]=0;
	if(!_order[DAFidx].empty())	{
		feature=_order[DAFidx][0];
		assert(feature>=0 && feature<_n);
		value=_stats[feature].daf[DAFidx];
		_itersize[DAFidx]=1;
		return true;
	} else return false;
}

template<class RETURNTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET>
bool Result_Tracker_Feature_Stats<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET>::getNextDAF(RETURNTYPE &value, DIMTYPE &feature, const unsigned int DAFidx) const
{
	assert(0<=DAFidx && DAFidx<3);
	if(0<_itersize[DAFidx] && _itersize[DAFidx]<_order[DAFidx].size()) {
		feature=_order[DAFidx][_itersize[DAFidx]];
		assert(feature>=0 && feature<_n);
		value=_stats[feature].daf[DAFidx];
		++_itersize[DAFidx];
		return true;
	}
	return false;
}


} // namespace
#endif // FSTRESULTTRACKERFEATURESTATS_H ///:~
