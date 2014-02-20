/*
 * Bayes++ the Bayesian Filtering Library
 * Copyright (c) 2004 Michael Stevens
 * See accompanying Bayes++.htm for terms and conditions of use.
 *
 * $Id$
 */

/*
 * SLAM : Simultaneous Locatization and Mapping
 *  Kalman filter representing representation of SLAM
 */

		// Bayes++ Bayesian filtering schemes
#include "BayesFilter/bayesFlt.hpp"
		// Bayes++ SLAM
#include "SLAM.hpp"
#include "kalmanSLAM.hpp"
#include <iostream>
#include <boost/numeric/ublas/io.hpp>

namespace SLAM_filter
{

template <class Base>
inline void zero(FM::ublas::matrix_range<Base> A)
// Zero a matrix_range
{	// Note A cannot be a reference
	typedef typename Base::value_type Base_value_type;
	FM::noalias(A) = FM::ublas::scalar_matrix<Base_value_type>(A.size1(),A.size2(), Base_value_type());
}

Kalman_SLAM::Kalman_SLAM( Kalman_filter_generator& filter_generator ) :
	SLAM(),
	fgenerator(filter_generator),
	loc(0), full(0)
{
	nL = 0;
	nM = 0;
}

Kalman_SLAM::~Kalman_SLAM()
{
	fgenerator.dispose (loc);
	fgenerator.dispose (full);
}

void Kalman_SLAM::init_kalman (const FM::Vec& x, const FM::SymMatrix& X)
{
	// TODO maintain map states
	nL = x.size();
	nM = 0;
	if (loc) fgenerator.dispose (loc);
	if (full) fgenerator.dispose (full);
		// generate a location filter for prediction
	loc = fgenerator.generate(nL);
		// generate full filter
	full = fgenerator.generate(nL);
		// initialise location states
	full->x.sub_range(0,nL) = x;
	full->X.sub_matrix(0,nL,0,nL) = X;
	full->init();
}

void Kalman_SLAM::predict( BF::Linrz_predict_model& lpred )
{
		// extract location part of full
	loc->x = full->x.sub_range(0,nL);
	loc->X = full->X.sub_matrix(0,nL,0,nL);
		// predict location, independent of map
	loc->init();
	loc->predict (lpred);
	loc->update();
		// return location to full
	full->x.sub_range(0,nL) = loc->x;
	full->X.sub_matrix(0,nL,0,nL) = loc->X;
	full->init();
}

void Kalman_SLAM::observe( unsigned feature, const Feature_observe& fom, const FM::Vec& z )
{
	// Assume features added sequentially
	if (feature >= nM) {
		error (BF::Logic_exception("Observe non existing feature"));
		return;
	}
	// TODO Implement nonlinear form
	// Create a augmented sparse observe model for full states
	BF::Linear_uncorrelated_observe_model fullm(full->x.size(), 1);
	fullm.Hx.clear();
	fullm.Hx.sub_matrix(0,nL, 0,nL) = fom.Hx.sub_matrix(0,nL, 0,nL);
	fullm.Hx(0,nL+feature) = fom.Hx(0,nL);
	fullm.Zv = fom.Zv;
	full->observe(fullm, z);
}

void Kalman_SLAM::observe_new( unsigned feature, const Feature_observe_inverse& fom, const FM::Vec& z )
// fom: must have a the special form required for SLAM::obeserve_new
{
		// size consistency, single state feature
	if (fom.Hx.size1() != 1)
		error (BF::Logic_exception("observation and model size inconsistent"));
		
		// make new filter with additional (uninitialized) feature state
	if (feature >= nM)
	{
		nM = feature+1;	
		Kalman_filter_generator::Filter_type* nf = fgenerator.generate(nL+nM);
		FM::noalias(nf->x.sub_range(0,full->x.size())) = full->x;
		FM::noalias(nf->X.sub_matrix(0,full->x.size(),0,full->x.size())) = full->X;

		fgenerator.dispose(full);
		full = nf;
	}
		// build augmented location and observation
	FM::Vec sz(nL+z.size());
	sz.sub_range(0,nL) = full->x.sub_range(0,nL);
	sz.sub_range(nL,nL+z.size() )= z;

	// TODO use named references rather then explict Ha Hb
	FM::Matrix Ha (fom.Hx.sub_matrix(0,1, 0,nL) );
	FM::Matrix Hb (fom.Hx.sub_matrix(0,1, nL,nL+z.size()) );
	FM::Matrix tempHa (1,nL);
	FM::Matrix tempHb (1,sz.size());

		// feature covariance with existing location and features
        // X+ = [0 Ha] X [0 Ha]' + Hb Z Hb'
        // - zero existing feature covariance
	zero( full->X.sub_matrix(0,full->X.size1(), nL+feature,nL+feature+1) );
	full->X.sub_matrix(nL+feature,nL+feature+1,0,nL+nM) = FM::prod(Ha,full->X.sub_matrix(0,nL, 0,nL+nM) );
		// feature state and variance
	full->x[nL+feature] = fom.h(sz)[0];
	full->X(nL+feature,nL+feature) = ( FM::prod_SPD(Ha,full->X.sub_matrix(0,nL, 0,nL),tempHa) +
													  FM::prod_SPD(Hb,fom.Zv,tempHb)
													 ) (0,0);
		
	full->init ();
}

void Kalman_SLAM::observe_new( unsigned feature, const FM::Float& t, const FM::Float& T )
{
		// Make space in scheme for feature, requires the scheme can deal with resized state
	if (feature >= nM)
	{
		Kalman_filter_generator::Filter_type* nf = fgenerator.generate(nL+feature+1);
		FM::noalias(nf->x.sub_range(0,full->x.size())) = full->x;
		FM::noalias(nf->X.sub_matrix(0,full->x.size(),0,full->x.size())) = full->X;
		zero( nf->X.sub_matrix(0,nf->X.size1(), nL+nM,nf->X.size2()) );

		nf->x[nL+feature] = t;
		nf->X(nL+feature,nL+feature) = T;
		nf->init ();
		fgenerator.dispose(full);
		full = nf;
		nM = feature+1;
	}
	else
	{
		full->x[nL+feature] = t;
		full->X(nL+feature,nL+feature) = T;
		full->init ();
	}
}

void Kalman_SLAM::forget( unsigned feature, bool must_exist )
{
	full->x[nL+feature] = 0.;
			// ISSUE uBLAS has problems accessing the lower symmetry via a sub_matrix proxy, there two two parts seperately
	zero( full->X.sub_matrix(0,nL+feature, nL+feature,nL+feature+1) );
	zero( full->X.sub_matrix(nL+feature,nL+feature+1, nL+feature,full->X.size1()) );
	full->init();
}

void Kalman_SLAM::decorrelate( Bayesian_filter::Bayes_base::Float d )
// Reduce correlation by scaling cross-correlation terms
{
	std::size_t i,j;
	const std::size_t n = full->X.size1();
	for (i = 1; i < n; ++i)
	{
		FM::SymMatrix::Row Xi(full->X,i);
		for (j = 0; j < i; ++j)
		{
			Xi[j] *= d;
		}
		for (j = i+1; j < n; ++j)
		{
			Xi[j] *= d;
		}
	}
	full->init();
}

}//namespace SLAM
