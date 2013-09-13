//#include "stdafx.h"
#include "error.hpp"
#include "global.hpp"
#include "subset.hpp"

#include "data_intervaller.hpp"
#include "data_splitter.hpp"
#include "data_splitter_5050.hpp"
//#include "data_splitter_cv.hpp"
//#include "data_splitter_holdout.hpp"
//#include "data_splitter_leave1out.hpp"
//#include "data_splitter_resub.hpp"
//#include "data_splitter_randrand.hpp"
//#include "data_splitter_randfix.hpp"
#include "data_scaler.hpp"
#include "data_scaler_void.hpp"
//#include "data_scaler_to01.hpp"
//#include "data_scaler_white.hpp"
#include "data_accessor_splitting_memTRN.hpp"
#include "data_accessor_splitting_memARFF.hpp"

//#include "criterion_normal_bhattacharyya.hpp"
#include "criterion_normal_gmahalanobis.hpp"
//#include "criterion_normal_divergence.hpp"
//#include "criterion_multinom_bhattacharyya.hpp"
//#include "criterion_wrapper.hpp"
//#include "criterion_wrapper_bias_estimate.hpp"
//#include "criterion_subsetsize.hpp"
//#include "criterion_sumofweights.hpp"
//#include "criterion_negative.hpp"

#include "distance_euclid.hpp"
//#include "distance_L1.hpp"
//#include "distance_Lp.hpp"
#include "classifier_knn.hpp"
//#include "classifier_normal_bayes.hpp"
//#include "classifier_multinom_naivebayes.hpp"
//#include "classifier_svm.hpp"

//#include "search_bif.hpp"
//#include "search_bif_threaded.hpp"
//#include "search_monte_carlo.hpp"
//#include "search_monte_carlo_threaded.hpp"
//#include "search_exhaustive.hpp"
//#include "search_exhaustive_threaded.hpp"
//#include "branch_and_bound_predictor_averaging.hpp"
//#include "search_branch_and_bound_basic.hpp"
//#include "search_branch_and_bound_improved.hpp"
//#include "search_branch_and_bound_partial_prediction.hpp"
//#include "search_branch_and_bound_fast.hpp"
#include "seq_step_straight.hpp"
//#include "seq_step_straight_threaded.hpp"
//#include "seq_step_hybrid.hpp"
//#include "seq_step_ensemble.hpp"
#include "search_seq_sfs.hpp"
//#include "search_seq_sffs.hpp"
//#include "search_seq_sfrs.hpp"
//#include "search_seq_os.hpp"
//#include "search_seq_dos.hpp"
//#include "result_tracker_dupless.hpp"
//#include "result_tracker_regularizer.hpp"
//#include "result_tracker_feature_stats.hpp"
//#include "result_tracker_stabileval.hpp"

#include <boost/smart_ptr.hpp>
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_fst {

// [ref] ${FST_HOME}/demo10.cpp
void demo_10()
{
	typedef double RETURNTYPE;
	typedef double DATATYPE;
	typedef double REALTYPE;

	typedef unsigned int IDXTYPE;
	typedef unsigned int DIMTYPE;
	typedef short BINTYPE;

	typedef FST::Subset<BINTYPE, DIMTYPE> SUBSET;
	typedef FST::Data_Intervaller<std::vector<FST::Data_Interval<IDXTYPE> >, IDXTYPE> INTERVALLER;
	typedef boost::shared_ptr<FST::Data_Splitter<INTERVALLER, IDXTYPE> > PSPLITTER;
	typedef FST::Data_Splitter_5050<INTERVALLER, IDXTYPE> SPLITTER5050;
	typedef FST::Data_Accessor_Splitting_MemTRN<DATATYPE, IDXTYPE, INTERVALLER> DATAACCESSOR;
	typedef FST::Criterion_Normal_GMahalanobis<RETURNTYPE, DATATYPE, REALTYPE, IDXTYPE, DIMTYPE, SUBSET, DATAACCESSOR> FILTERCRIT;
	typedef FST::Sequential_Step_Straight<RETURNTYPE, DIMTYPE, SUBSET, FILTERCRIT> EVALUATOR;
	typedef FST::Distance_Euclid<DATATYPE, DIMTYPE, SUBSET> DISTANCE;
	typedef FST::Classifier_kNN<RETURNTYPE, DATATYPE, IDXTYPE, DIMTYPE, SUBSET, DATAACCESSOR, DISTANCE> CLASSIFIERKNN;

	std::cout << "Starting Example 10: Basic Filter-based feature selection..." << std::endl;

	// in the course of search use the first half of data for feature selection and the second half for testing using 3-NN classifier
	PSPLITTER dsp(new SPLITTER5050());

	// do not scale data
	boost::shared_ptr<FST::Data_Scaler<DATATYPE> > dsc(new FST::Data_Scaler_void<DATATYPE>());

	// set-up data access
	boost::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>); 
	splitters->push_back(dsp);
	boost::shared_ptr<DATAACCESSOR> da(new DATAACCESSOR("./data/feature_analysis/fst/speech_15.trn", splitters, dsc));
	da->initialize();

	// initiate access to split data parts
	da->setSplittingDepth(0);
	if (!da->getFirstSplit())
		throw FST::fst_error("50/50 data split failed.");

	// initiate the storage for subset to-be-selected
	boost::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures()));
	sub->deselect_all();

	// set-up the normal Generalized Mahalanobis criterion 
	boost::shared_ptr<FILTERCRIT> crit(new FILTERCRIT);
	crit->initialize(da);  // initialization = normal model parameter estimation on training data part

	// set-up the standard sequential search step object (options: hybrid, ensemble, etc.)
	boost::shared_ptr<EVALUATOR> eval(new EVALUATOR);

	// set-up Sequential Forward Floating Selection search procedure
	FST::Search_SFS<RETURNTYPE,DIMTYPE, SUBSET, FILTERCRIT, EVALUATOR> srch(eval);
	srch.set_search_direction(FST::FORWARD);  // try FST::BACKWARD
	
	// run the search
	std::cout << "Feature selection setup:" << std::endl << *da << std::endl << srch << std::endl << *crit << std::endl << std::endl;
	RETURNTYPE critval_train, critval_test;
	const DIMTYPE d = 7;  // request subset of size d; if set to 0, cardinality will decided in the course of search
	if (!srch.search(d, critval_train, sub, crit, std::cout))
		throw FST::fst_error("Search not finished.");
	
	// (optionally) the following line is included here just for illustration because srch.search() reports results in itself
	std::cout << std::endl << "Search result: " << std::endl << *sub << std::endl << "Criterion value = " << critval_train << std::endl << std::endl;
	
	// (optionally) validate result by estimating kNN accuracy on selected feature sub-space on independent test data
	boost::shared_ptr<CLASSIFIERKNN> cknn(new CLASSIFIERKNN);
	cknn->set_k(3);

	da->setSplittingDepth(0);
	cknn->train(da,sub);
	cknn->test(critval_test,da);

	std::cout << "Validated " << cknn->get_k() << "-NN accuracy = " << critval_test << std::endl << std::endl;
}

}  // namespace my_fst
