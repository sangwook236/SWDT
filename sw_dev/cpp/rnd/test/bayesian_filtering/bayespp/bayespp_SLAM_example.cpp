#include <map>
#include <BayesFilter/SIRFlt.hpp>
#include <BayesFilter/covFlt.hpp>
#include <BayesFilter/unsFlt.hpp>
#include <BayesFilter/models.hpp>
#include <SLAM/SLAM.hpp>
#include <SLAM/fastSLAM.hpp>
#include <SLAM/kalmanSLAM.hpp>
#include <Test/random.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>
#include <vector>


namespace {
namespace local {

// random numbers for SLAM test.
class SLAM_random : public Bayesian_filter_test::Boost_random, public Bayesian_filter::SIR_random
{
public:
	Bayesian_filter_matrix::Float normal(const Bayesian_filter_matrix::Float mean, const Bayesian_filter_matrix::Float sigma)
	{
		return Boost_random::normal(mean, sigma);
	}
	void normal(Bayesian_filter_matrix::DenseVec &v)
	{
		Boost_random::normal(v);
	}
	void uniform_01(Bayesian_filter_matrix::DenseVec &v)
	{
		Boost_random::uniform_01(v);
	}
	void seed()
	{
		Boost_random::seed();
	}
};

// demonstrate a SLAM example.
struct SLAMDemo
{
public:	
	SLAMDemo(unsigned int setnParticles) : nParticles(setnParticles)
	{}

public:	
	void OneDExperiment();
	void InformationLossExperiment();
	
	// relative observation with noise model.
	struct Simple_observe : Bayesian_filter::Linear_uncorrelated_observe_model
	{
		// construct a linear model with const Hx.
		Simple_observe(Bayesian_filter_matrix::Float i_Zv) : Linear_uncorrelated_observe_model(2, 1)
		{
			Hx(0, 0) = -1.0;  // location.
			Hx(0, 1) = 1.0;  // map.
			Zv[0] = i_Zv;
		}
	};
	struct Simple_observe_inverse : Bayesian_filter::Linear_uncorrelated_observe_model
	{
		Simple_observe_inverse(Bayesian_filter_matrix::Float i_Zv) : Linear_uncorrelated_observe_model(2, 1)
		{
			Hx(0, 0) = 1.0;  // location.
			Hx(0, 1) = 1.0;  // observation.
			Zv[0] = i_Zv;
		}
	};

	// Kalman statistics without any filtering.
	struct Kalman_statistics : public Bayesian_filter::Kalman_state_filter
	{
		Kalman_statistics(std::size_t x_size) : Bayesian_filter::Kalman_state_filter(x_size) {}
		void init() {}
		void update() {}
	};

	// generate and dispose of generic Kalman filter type.
	template<class Filter>
	struct Generic_kalman_generator : public SLAM_filter::Kalman_filter_generator
	{
		SLAM_filter::Kalman_filter_generator::Filter_type * generate(unsigned full_size)
		{
			return new Filter(full_size);
		}
		void dispose(SLAM_filter::Kalman_filter_generator::Filter_type *filter)
		{
			delete filter;
		}
	};

	void display(const std::string label, const Bayesian_filter::Kalman_state_filter &stats)
	{
		std::cout << label << stats.x << stats.X << std::endl;
	}

private:
	const unsigned int nParticles;
	SLAM_random goodRandom;
};

// experiment with a one dimensional problem.
// use to look at implication of highly correlated features.
void SLAMDemo::OneDExperiment()
{
	// state size.
	const unsigned int nL = 1;  // location.
	const unsigned int nM = 2;  // map.

	// construct simple Prediction models.
	Bayesian_filter::Sampled_LiAd_predict_model location_predict(nL, 1, goodRandom);
	// stationary prediction model (identity).
	Bayesian_filter_matrix::identity(location_predict.Fx);
	// constant noise model.
	location_predict.q[0] = 1000.0;
	location_predict.G.clear();
	location_predict.G(0, 0) = 1.0;

	// relative observation with noise model.
	Simple_observe observe0(5.0), observe1(3.0);
	Simple_observe_inverse observe_new0(5.0), observe_new1(3.0);

	// setup the initial state and covariance location with no uncertainty.
	Bayesian_filter_matrix::Vec x_init(nL);
	Bayesian_filter_matrix::SymMatrix X_init(nL, nL);
	x_init[0] = 20.0;
	X_init(0, 0) = 0.0;

	// truth model : location plus one map feature.
	Bayesian_filter_matrix::Vec true0(nL + 1), true1(nL + 1);
	true0.sub_range(0, nL) = x_init;
	true0[nL] = 50.0;
	true1.sub_range(0, nL) = x_init;
	true1[nL] = 70.0;
	Bayesian_filter_matrix::Vec z(1);

	// filter statistics for display.
	Kalman_statistics stat(nL + nM);

	// Kalman SLAM filter.
	Generic_kalman_generator<Bayesian_filter::Covariance_scheme> full_gen;
	SLAM_filter::Kalman_SLAM kalm(full_gen);
	kalm.init_kalman(x_init, X_init);

	// fast SLAM filter.
	Bayesian_filter::SIR_kalman_scheme fast_location(nL, nParticles, goodRandom);
	fast_location.init_kalman(x_init, X_init);
	SLAM_filter::Fast_SLAM_Kstatistics fast(fast_location);

	// initial feature states.
	z = observe0.h(true0);  // observe a relative position between location and map landmark.
	z[0] += 0.5;			
	kalm.observe_new(0, observe_new0, z);
	fast.observe_new(0, observe_new0, z);

	z = observe1.h(true1);
	z[0] += -1.0;		
	kalm.observe_new(1, observe_new1, z);
	fast.observe_new(1, observe_new1, z);

	fast.update();  fast.statistics_sparse(stat);  display("Feature Fast", stat);
	kalm.update();  kalm.statistics_sparse(stat);  display("Feature Kalm", stat);

	// predict the location state forward.
	fast_location.predict (location_predict);
	kalm.predict (location_predict);
	fast.update();  fast.statistics_sparse(stat);  display("Predict Fast", stat);
	kalm.update();  kalm.statistics_sparse(stat);  display("Predict Kalm", stat);

	// observation feature 0.
	z = observe0.h(true0);
	z[0] += 0.5;  // observe a relative position between location and map landmark.
	fast.observe(0, observe0, z);
	kalm.observe(0, observe0, z);
	fast.update();  fast.statistics_sparse(stat);  display("ObserveA Fast", stat);
	kalm.update();  kalm.statistics_sparse(stat);  display("ObserveA Kalm", stat);

	// observation feature 1.
	z = observe1.h(true1);
	z[0] += 1.0;  // observe a relative position between location and map landmark.
	fast.observe(1, observe1, z);
	kalm.observe(1, observe1, z);
	fast.update();  fast.statistics_sparse(stat);  display("ObserveB Fast", stat);
	kalm.update();  kalm.statistics_sparse(stat);  display("ObserveB Kalm", stat);

	// observation feature 0.
	z = observe0.h(true0);
	z[0] += 0.5;  // observe a relative position between location and map landmark.
	fast.observe(0, observe0, z);
	kalm.observe(0, observe0, z);
	fast.update();  fast.statistics_sparse(stat);  display("ObserveC Fast", stat);
	kalm.update();  kalm.statistics_sparse(stat);  display("ObserveC Kalm", stat);

	// forget feature 0.
	fast.forget(0);
	kalm.forget(0);
	fast.update();  fast.statistics_sparse(stat);  display("Forget Fast", stat);
	kalm.update();  kalm.statistics_sparse(stat);  display("Forget Kalm", stat);
}

// experiment with information loss due to resampling.
void SLAMDemo::InformationLossExperiment()
{
	// state size.
	const unsigned int nL = 1;  // location.
	const unsigned int nM = 2;  // map.

	// construct simple Prediction models.
	Bayesian_filter::Sampled_LiAd_predict_model location_predict(nL, 1, goodRandom);
	// stationary Prediction model (identity).
	Bayesian_filter_matrix::identity(location_predict.Fx);
	// constant noise model.
	location_predict.q[0] = 1000.0;
	location_predict.G.clear();
	location_predict.G(0, 0) = 1.0;

	// relative observation with noise model.
	Simple_observe observe0(5.0), observe1(3.0);
	Simple_observe_inverse observe_new0(5.0), observe_new1(3.0);

	// setup the initial state and covariance location with no uncertainty.
	Bayesian_filter_matrix::Vec x_init(nL);
	Bayesian_filter_matrix::SymMatrix X_init(nL, nL);
	x_init[0] = 20.0;
	X_init(0, 0) = 0.0;

	// truth model : location plus one map feature.
	Bayesian_filter_matrix::Vec true0(nL + 1), true1(nL + 1);
	true0.sub_range(0, nL) = x_init;
	true0[nL] = 50.0;
	true1.sub_range(0, nL) = x_init;
	true1[nL] = 70.0;
	Bayesian_filter_matrix::Vec z(1);

	// filter statistics for display.
	Kalman_statistics stat(nL + nM);

	// Kalman SLAM filter.
	Generic_kalman_generator<Bayesian_filter::Unscented_scheme> full_gen;
	SLAM_filter::Kalman_SLAM kalm(full_gen);
	kalm.init_kalman(x_init, X_init);

	// fast SLAM filter.
	Bayesian_filter::SIR_kalman_scheme fast_location(nL, nParticles, goodRandom);
	fast_location.init_kalman(x_init, X_init);
	SLAM_filter::Fast_SLAM_Kstatistics fast(fast_location);

	// initial feature states.
	z = observe0.h(true0);  // observe a relative position between location and map landmark.
	z[0] += 0.5;			
	kalm.observe_new(0, observe_new0, z);
	fast.observe_new(0, observe_new0, z);

	z = observe1.h(true1);
	z[0] += -1.0;		
	kalm.observe_new(1, observe_new1, z);
	fast.observe_new(1, observe_new1, z);

	unsigned int it = 0;
	for (;;)
	{
		++it;
		std::cout << it << std::endl;
		
		// groups of observations without resampling.
		{
			// predict the filter forward.
			kalm.predict(location_predict);
			fast_location.predict(location_predict);

			// observation feature 0 with bias.
			z = observe0.h(true0);  // observe a relative position between location and map landmark.
			z[0] += 0.5;
			kalm.observe(0, observe0, z);
			fast.observe(0, observe0, z);

			// predict the filter forward.
			kalm.predict(location_predict);
			fast_location.predict(location_predict);

			// observation feature 1 with bias.
			z = observe1.h(true1);  // observe a relative position between location and map landmark.
			z[0] += -1.0;		
			kalm.observe(1, observe1, z);
			fast.observe(1, observe1, z);
		}

		// update and resample.
		kalm.update();
		fast.update();
		
		kalm.statistics_sparse(stat); display("Kalm", stat);
		fast.statistics_sparse(stat); display("Fast", stat);
		std::cout << fast_location.stochastic_samples << ',' << fast_location.unique_samples()
			<< ' ' << fast.feature_unique_samples(0) << ',' << fast.feature_unique_samples(1) << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_bayespp {

// ${BAYES++_HOME}/SLAM/testFastSLAM.cpp.
void SLAM_example()
{
	// global setup for test output.
	std::cout.flags(std::ios::fixed);
	std::cout.precision(4);

	const unsigned int nParticles = 1000;
	std::cout << "nParticles = " << nParticles << std::endl;

	// create test and run experiments.
	try
	{
		local::SLAMDemo test(nParticles);
		test.OneDExperiment();
		//test.InformationLossExperiment();
	}
	catch (const Bayesian_filter::Filter_exception &e)
	{
		std::cerr << "Bayesian_filter::Filter_exception caught: " << e.what() << std::endl;
	}
}

}  // namespace my_bayespp
