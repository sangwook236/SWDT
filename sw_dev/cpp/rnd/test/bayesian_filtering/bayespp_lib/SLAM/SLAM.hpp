/*
 * Bayes++ the Bayesian Filtering Library
 * Copyright (c) 2004 Michael Stevens
 * See accompanying Bayes++.htm for terms and conditions of use.
 *
 * $Id$
 */

/*
 * SLAM : Simultaneous Locatization and Mapping
 */

namespace SLAM_filter
{
namespace BF = Bayesian_filter;
namespace FM = Bayesian_filter_matrix;

class SLAM : public BF::Bayes_filter_base
/*
 * SLAM : Simultaneous Location and Mapping
 *  Abstract representation of general SLAM
 * The abstraction represents the feature observe functions. The observe functions
 * depend on current location and map feature states. Map features are defined as scalers.
 * Multiple features must be used to represent vector map states.
 *
 * Observe function parameters are defined:
 *   feature: a arbitrary unique number to label each feature in the map
 *   fom: feature observe model
 *   z: observation vector
 *
 * A complete SLAM solution must also represent location and map predict functions.
 * This is not include in the abstraction as no single implementation can deal with
 * a general stochastic predict model.
 */
{
public:
	SLAM ()
	{}

									// Observation models
	typedef BF::Linrz_uncorrelated_observe_model Feature_observe;
	// Linearised observation model
	//  Observation z = h(lt) where lt is the vector of location state augmented with the associated feature state

	typedef BF::Linrz_uncorrelated_observe_model Feature_observe_inverse;
	// Inverse model required for observe_new
	//  Feature state t = h(lz)	where lz is the vector of location state augmented with observation z

									// Observation associated with a single feature
	virtual void observe( unsigned feature, const Feature_observe& fom, const FM::Vec& z ) = 0;
	// Feature observation (fuse with existing feature)
	virtual void observe_new( unsigned feature, const Feature_observe_inverse& fom, const FM::Vec& z ) = 0;
	// New Feature observation (new or overwrite existing feature)
	virtual void observe_new( unsigned feature, const FM::Float& t, const FM::Float& T ) = 0;
	// New Feature directly from Feature statistics: mean t and variance T (overwrite existing feature)

	virtual void forget( unsigned feature, bool must_exist = true ) = 0;
	// Forget information associated with a feature: feature number can be reused for a new feature

										// Observation associated with multiple features
                                                // use a vector to store multiple feature associations
                                                // the 'fom' is constructed to conform with feature state stacked in the order they appear in 'features'
	typedef std::vector<unsigned> Multi_features_t;
	virtual void multi_observe( const Multi_features_t& features, const Feature_observe& fom, const FM::Vec& z )
	{
		error (BF::Logic_exception("Unimplemented"));
	}
	virtual void multi_observe_new( const Multi_features_t& features, const Feature_observe_inverse& fom, const FM::Vec& z )
	{
		error (BF::Logic_exception("Unimplemented"));
	}
	virtual void multi_observe_new( const Multi_features_t& features, const FM::Vec& t, const FM::Vec& T )
	{
		error (BF::Logic_exception("Unimplemented"));
	}
	virtual void multi_forget( const Multi_features_t& features, bool must_exist = true )
	{
		error (BF::Logic_exception("Unimplemented"));
	}

                                        // Observation associated with all features (unassociated)
                                                // the 'fom' is constructed to conform with feature state stacked in feature number order
    virtual void all_observe( const Feature_observe& fom, const FM::Vec& z )
   {
      error (BF::Logic_exception("Unimplemented"));
  }
  virtual void all_observe_new( const Feature_observe_inverse& fom, const FM::Vec& z )
   {
      error (BF::Logic_exception("Unimplemented"));
  }
  virtual void all_observe_new( const FM::Vec& t, const FM::Vec& T )
 {
      error (BF::Logic_exception("Unimplemented"));
  }
  virtual void all_forget( )
  {
      error (BF::Logic_exception("Unimplemented"));
  }

	virtual void update () = 0;
	// Update SLAM state after a sequence of observe or forget
};


}//namespace SLAM
