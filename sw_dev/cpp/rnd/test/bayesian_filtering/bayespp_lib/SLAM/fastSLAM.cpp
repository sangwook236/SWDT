/*
 * Bayes++ the Bayesian Filtering Library
 * Copyright (c) 2004 Michael Stevens
 * See accompanying Bayes++.htm for terms and conditions of use.
 *
 * $Id$
 */

/*
 * SLAM : Simultaneous Localisation and Mapping
 *  FastSLAM augmented particle algorithm
 *  Direct implementation without log(n) tree pruning for associated observations
 */

		// Bayes++ Bayesian filtering schemes
#include "BayesFilter/SIRFlt.hpp"
		// Types required for SLAM classes
#include <map>
		// Bayes++ SLAM
#include "SLAM.hpp"
#include "fastSLAM.hpp"
#include <cmath>


namespace {
	template <class scalar>
	inline scalar sqr(scalar x)
	{
		return x*x;
	}
}

namespace SLAM_filter
{

Fast_SLAM::Fast_SLAM( BF::SIR_scheme& L_filter ) :
	SLAM(),
	L(L_filter),
	wir(L.S.size2())
// Construct filter using referenced SIR_filter for resampling
{
	std::fill (wir.begin(), wir.end(), Float(1.));		// Initial uniform weights
	wir_update = false;
}

void Fast_SLAM::observe_new( unsigned feature, const Feature_observe_inverse& fom, const FM::Vec& z )
/*
 * SLAM New Feature observation (overwrite)
 * Assumes there is no prior information about the feature (strictly a uniform un-informative prior)
 *
 * This implies
 *  a) There has no information about location so no resampling is requires
 *  b) Feature Posterior estimated directly from observation using information form
 *
 * fom: must have a the special from required for SLAM::obeserve_new
 */
{
	assert(z.size() == 1);		// Only single state observation supported

	const std::size_t nL = L.S.size1();	// No of location states
	const std::size_t nparticles = L.S.size2();
	FeatureCondMap fmap(nparticles);

	FM::Vec sz(nL+1);						// Location state

	for (std::size_t pi = 0; pi < nparticles; ++pi)
	{
		sz.sub_range(0,nL) = FM::column (L.S, pi);
		sz[nL] = z[0];
		FM::Vec t = fom.h(sz);
		fmap[pi].x = t[0];
		fmap[pi].X = fom.Zv[0];
	}
	M.insert (std::make_pair(feature, fmap));
}

void Fast_SLAM::observe_new( unsigned feature, const FM::Float& t, const FM::Float& T )
/*
 * SLAM New observation directly of state statistics (overwrite)
 */
{
	const std::size_t nparticles = L.S.size2();
	Feature_1 m1;  				// single map feature
	FeatureCondMap fmap(nparticles);

	m1.x = t;			// Initial particle conditional map is sample
	m1.X = T;		    // Independent
	std::fill(fmap.begin(),fmap.end(), m1);
	M.insert (std::make_pair(feature, fmap));
}

void Fast_SLAM::observe( unsigned feature, const Feature_observe& fom, const FM::Vec& z )
/*
 * SLAM Feature observation
 *  Uses Extended Fast_SLAM observation equations
 * Note: Mathematically only weight ratios are important. Numerically however the range should be restricted.
 * The weights are computed here using the simplest form with common factor Ht removed.
 */
{
	assert(z.size() == 1);		// Only single state observation supported

	const AllFeature::iterator inmap = M.find(feature);
	if (inmap == M.end())
	{
		error (BF::Logic_exception("Observe non existing feature"));
		return;
	}
								// Existing feature
	FeatureCondMap& afm = (*inmap).second;	// Reference the associated feature map
	const std::size_t nL = L.S.size1();	// No of location states
	const std::size_t nparticles = L.S.size2();

	Float Ht = fom.Hx(0,nL);
	if (Ht == 0)
		error (BF::Numeric_exception("observe Hx feature component zero"));

							// Loop in invariants and temporary storage
	FM::Vec x2(nL+1);					// Augmented state (particle + feature mean)
	FM::Vec znorm(z.size());
	const Float Z = fom.Zv[0];
							// Iterate over particles
	for (std::size_t pi = 0; pi != nparticles; ++pi)
	{
		Feature_1& m1 = afm[pi];		// Associated feature's map particle
							
		x2.sub_range(0,nL) = FM::column (L.S, pi);		// Build Augmented state x2
		x2[nL] = m1.x;
		const FM::Vec& zp = fom.h(x2);	// Observation model
		znorm = z;									// Normalised observation
		fom.normalise(znorm, zp);
														// Observation innovation and innovation variance
		const Float s = (znorm[0] - zp[0]);
		const Float S = sqr(Ht) * m1.X + Z;
		if (S <= 0)
			error (BF::Numeric_exception("Conditional feature estimate not PD"));
		const Float sqrtS  = std::sqrt (S);		// Drop Ht which is a common factor for all weights

										// Multiplicative fusion of observation weights, integral of Gaussian product g(p,P)*g(q,Q)
 		wir[pi] *= exp(Float(-0.5)* sqr(s) / S) / sqrtS;

										// Estimate associated features conditional map for resampled particles
		Float W = m1.X*Ht / S;	// EKF for conditional feature observation - specialised for 1D and zero state uncertianty
		m1.x += W * (znorm[0] - zp[0]);
		m1.X -= sqr(W) * S;
	}
	wir_update = true;			// Weights have been updated requiring a resampling
}

Fast_SLAM::Float
 Fast_SLAM::update_resample( const Bayesian_filter::Importance_resampler& resampler )
/* Resampling Update
 *  Resample particles using weights
 *  Propagate resampling to All features
 *  Only resamples if weights have been updated
 */
{
	if (wir_update)
	{
		const std::size_t nparticles = L.S.size2();
		Resamples_t presamples(nparticles);
		std::size_t R_unique;			// Determine resamples of S
		Float lcond = resampler.resample (presamples, R_unique, wir, L.random);

									// Initial uniform weights
		std::fill (wir.begin(), wir.end(), Float(1.));
		wir_update = false;
									// Update S bases on resampling, and init filter
		L.copy_resamples (L.S, presamples);
		L.init_S ();
									// Propagate resampling to All features
		FeatureCondMap fmr(nparticles);		// Resampled feature map
		for (AllFeature::iterator fi = M.begin(); fi != M.end(); ++fi)	// All Features
		{
			FeatureCondMap& fm = (*fi).second;		// Reference the feature map
										// Iterate over All feature particles
			FeatureCondMap::iterator fmi, fmi_begin = fm.begin(), fmi_end = fm.end();
			FeatureCondMap::iterator fmri = fmr.begin();
			for (fmi = fmi_begin; fmi < fmi_end; ++fmi)
			{							// Multiple copies of this resampled feature
				for (std::size_t res = presamples[fmi-fmi_begin]; res > 0; --res) {
					*fmri = *fmi;
					++fmri;
				}
			}
			fm = fmr;				// Copy in resamples feature map
		}

		L.roughen ();				// Roughen location
		L.stochastic_samples = R_unique;
		return lcond;
	}
	else
		return 1.;		// No resampling
}

void Fast_SLAM::forget( unsigned feature, bool must_exist )
// Forget all feature information, feature no can be reused for a new feature
{
	AllFeature::size_type n = M.erase(feature);
	if (n == 0 && must_exist)
		error (BF::Logic_exception("Forget non existing feature"));
}

std::size_t Fast_SLAM::feature_unique_samples( unsigned feature )
/*
 * Count the number of unique samples in S associated with a feature
 */
{
	const AllFeature::iterator inmap = M.find(feature);
	if (inmap == M.end())
	{
		error (BF::Logic_exception("feature_unique_samples non existing feature"));
		return 0;
	}
								// Existing feature
	FeatureCondMap& afm = (*inmap).second;	// Reference the associated feature map

	typedef FeatureCondMap::iterator Sref;
	// Provide a ordering on feature sample means
	struct order {
		static bool less(Sref a, Sref b)
		{
			return (*a).x < (*b).x;
		}
	};

						// Sorted reference container
	typedef std::vector<Sref> SRContainer;
	SRContainer sortR(afm.size());

						// Reference each element in S
	{	Sref elem = afm.begin();
		SRContainer::iterator ssi = sortR.begin();
		for (; ssi < sortR.end(); ++ssi)
		{
			*ssi = elem; ++elem;
		}
	}

	std::sort (sortR.begin(), sortR.end(), order::less);

						// Count element changes, precond: sortS not empty
	std::size_t u = 1;
	SRContainer::const_iterator ssi= sortR.begin();
	SRContainer::const_iterator ssp = ssi;
	++ssi;
	while (ssi < sortR.end())
	{
		if (order::less(*ssp, *ssi))
			++u;
		ssp = ssi;
		++ssi;
	}
	return u;
}


/*
 * Fast_SLAM_Kstatistics
 */
Fast_SLAM_Kstatistics::Fast_SLAM_Kstatistics( BF::SIR_kalman_scheme& L_filter ) :
	Fast_SLAM(L_filter), L(L_filter)
// Construct filter using referenced SIR_filter for resampling
{
}


void Fast_SLAM_Kstatistics::statistics_feature(
		BF::Kalman_state_filter& kstat, std::size_t fs,
		const AllFeature::const_iterator& fi, const AllFeature::const_iterator& fend ) const
/*
 * Compute sample mean and covariance statistics of feature
 * We use the Maximum Likelihood (bias) estimate definition of covariance (1/n)
 *  fs is subscript in kstat to return statisics
 *  fi iterator of feature
 *  fend end of map iterator (statistics are computed for fi with a map subset)
 *
 * Numerics
 *  No check is made for the conditioning of samples with regard to mean and covariance
 *  Extreme ranges or very large sample sizes will result in inaccuracy
 *  The covariance should always remain PSD however
 * 
 * Precond: kstat contains location mean
 */
{
	const std::size_t nL = L.S.size1();	// No of location states
	const std::size_t nparticles = L.S.size2();

	const FeatureCondMap& fm = (*fi).second;		// Reference the feature map

									// Iterate over All feature particles
	FeatureCondMap::const_iterator fpi, fpi_begin = fm.begin(), fpi_end = fm.end();

	Float mean_f = 0;				// Feature mean
	for (fpi = fpi_begin; fpi < fpi_end; ++fpi)	{
		mean_f += (*fpi).x;
	}
	mean_f /= Float(nparticles);

	Float var_f = 0;				// Feature variance: is ML estimate given estimated mean
	for (fpi = fpi_begin; fpi < fpi_end; ++fpi) {
		var_f += (*fpi).X + sqr((*fpi).x - mean_f);
	}
	var_f /= Float(nparticles);

	kstat.x[fs] = mean_f;			// Copy into Kalman statistics
	kstat.X(fs,fs) = var_f;

									// Location,feature covariance
	for (std::size_t si = 0; si < nL; ++si)
	{
		Float covar_f_si = 0;
		std::size_t spi = 0;
		const Float mean_si = kstat.x[si];
		for (fpi = fpi_begin; fpi < fpi_end; ++fpi, ++spi) {
			covar_f_si += ((*fpi).x - mean_f) * (L.S(si,spi) - mean_si);
		}
		covar_f_si /= Float(nparticles);
		kstat.X(si,fs) = covar_f_si;
	}

									// Feature,feature covariance. Iterate over previous features with means already computed
	std::size_t fsj = nL;				// Feature subscript
	for (AllFeature::const_iterator fj = M.begin(); fj != fend; ++fj, ++fsj)
	{
		Float covar_f_fi = 0;
		FeatureCondMap::const_iterator fpj = (*fj).second.begin();
		const Float mean_fi = kstat.x[fsj];
		for (fpi = fpi_begin; fpi < fpi_end; ++fpi) {
			covar_f_fi += ((*fpi).x - mean_f) * ((*fpj).x - mean_fi);
			++fpj;
		}
		covar_f_fi /= Float(nparticles);
		kstat.X(fs,fsj) = covar_f_fi;
	}
}


void Fast_SLAM_Kstatistics::statistics_compressed( BF::Kalman_state_filter& kstat )
/*
 * Compute sample mean and covariance statistics of filter
 *  
 *  kstat elements are filled first with Location statistics and then the Map feature statistics
 *  Feature statistics are are computed in feature number order and only for those for which there is space in kstat
 * Note: Covariance values are indeterminate for nparticles ==1
 * Precond:
 *   nparticles >=1 (enforced by Sample_filter construction)
 *   kstat must have space for Location statistics
 * Postcond:
 *  kstat compressed sample statistics of filter
 */
{	
	const std::size_t nL = L.S.size1();	// No of location states

	kstat.x.clear();			// Zero everything (required only for non existing feature states
	kstat.X.clear();			// Zero everything (required only for non existing feature states

								// Get Location statistics
	if (nL > kstat.x.size())
		error (BF::Logic_exception("kstat to small to hold filter location statistics"));
	L.update_statistics();
	FM::noalias(kstat.x.sub_range(0,nL)) = L.x;
	FM::noalias(kstat.X.sub_matrix(0,nL, 0,nL)) = L.X;

								// Iterated over feature statistics (that there is space for in kstat)
	std::size_t fs = nL;						// Feature subscript
	for (AllFeature::const_iterator fi = M.begin(); fi != M.end() && fs < kstat.x.size(); ++fi, ++fs)
	{
		statistics_feature(kstat, fs, fi, fi);	// build statistics of fi with other features up to fi
	}
}//statistics_compressed


void Fast_SLAM_Kstatistics::statistics_sparse( BF::Kalman_state_filter& kstat )
/*
 * Compute sample mean and covariance statistics of filter
 *  
 *  kstat elements are filled first with Location statistics and then the Map feature statistics
 *  Feature statistics are are computed in feature number as index (after location) and only for those for which there is space in kstat
 * Note: Covariance values are indeterminate for nparticles ==1
 * Precond:
 *   nparticles >=1 (enforced by Sample_filter construction)
 *   kstat must have space for Location statistics
 * Postcond:
 *  kstat sparse sample statistics of filter
 */
{	
	const std::size_t nL = L.S.size1();	// No of location states

	kstat.x.clear();			// Zero everything (required only for non existing feature states
	kstat.X.clear();			// Zero everything (required only for non existing feature states

								// Get Location statistics
	if (nL > kstat.x.size())
		error (BF::Logic_exception("kstat to small to hold filter location statistics"));
	L.update_statistics();
	FM::noalias(kstat.x.sub_range(0,nL)) = L.x;
	FM::noalias(kstat.X.sub_matrix(0,nL, 0,nL)) = L.X;

								// Iterated over feature statistics (that there is space for in kstat)
	for (AllFeature::const_iterator fi = M.begin(); fi != M.end(); ++fi)
	{
		std::size_t fs = nL + (*fi).first;		// Feature subscript
		if (fs < kstat.x.size())			// Space in kstat
		{
			statistics_feature(kstat, fs, fi, fi);	// build statistics of fi with other features up to fi
		}
	}//all feature
}//statistics_sparse


}//namespace SLAM
