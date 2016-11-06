#include <iostream>
#include <algorithm>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/array.hpp>
#include <boost/foreach.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>


namespace {
namespace local {

// REF [file] >> ${BOOST_HOME}/libs/accumulators/example/main.cpp
// Helper that uses BOOST_FOREACH to display a range of doubles.
template<typename Range>
void output_range(Range const &rng)
{
	bool first = true;
	BOOST_FOREACH(double d, rng)
	{
		if (!first) std::cout << ", ";
		std::cout << d;
		first = false;
	}
	std::cout << '\n';
}

// REF [file] >> ${BOOST_HOME}/libs/accumulators/example/main.cpp
void simple_example_1()
{
	boost::accumulators::accumulator_set<
		double,
		boost::accumulators::stats<boost::accumulators::tag::min, boost::accumulators::tag::mean(boost::accumulators::immediate), boost::accumulators::tag::sum, boost::accumulators::tag::moment<2> >
	> acc;

	boost::array<double, 4> data = { 0., 1., -1., 3.14159 };

	// std::for_each pushes each sample into the accumulator one at a time, and returns a copy of the accumulator.
	acc = std::for_each(data.begin(), data.end(), acc);

	// The following would be equivalent, and could be more efficient because it doesn't pass and return the entire accumulator set by value.
	//std::for_each(data.begin(), data.end(), boost::bind<void>(boost::ref(acc), _1));

	std::cout << "\tmin""(acc)        = " << (boost::accumulators::min)(acc) << std::endl;  // Extra quotes are to prevent complaints from Boost inspect tool.
	std::cout << "\tmean(acc)       = " << boost::accumulators::mean(acc) << std::endl;

	// Since mean depends on count and sum, we can get their results, too.
	std::cout << "\tcount(acc)      = " << boost::accumulators::count(acc) << std::endl;
	std::cout << "\tsum(acc)        = " << boost::accumulators::sum(acc) << std::endl;
	std::cout << "\tmoment<2>(acc)  = " << boost::accumulators::moment<2>(acc) << std::endl;  // A raw moment or crude moment.
}

// REF [file] >> ${BOOST_HOME}/libs/accumulators/example/main.cpp
void simple_example_2()
{
	// An accumulator which tracks the right tail (largest N items) and some data that are covariate with them. N == 4.
	boost::accumulators::accumulator_set<
		double,
		boost::accumulators::stats<boost::accumulators::tag::tail_variate<double, boost::accumulators::tag::covariate1, boost::accumulators::right> >
	> acc(boost::accumulators::tag::tail<boost::accumulators::right>::cache_size = 4);

	acc(2.1, boost::accumulators::covariate1 = .21);
	acc(1.1, boost::accumulators::covariate1 = .11);
	acc(2.1, boost::accumulators::covariate1 = .21);
	acc(1.1, boost::accumulators::covariate1 = .11);

	std::cout << "\ttail            = "; output_range(boost::accumulators::tail(acc));
	std::cout << "\ttail_variate    = "; output_range(boost::accumulators::tail_variate(acc));
	std::cout << std::endl;

	acc(21.1, boost::accumulators::covariate1 = 2.11);
	acc(11.1, boost::accumulators::covariate1 = 1.11);
	acc(21.1, boost::accumulators::covariate1 = 2.11);
	acc(11.1, boost::accumulators::covariate1 = 1.11);

	std::cout << "\ttail            = "; output_range(boost::accumulators::tail(acc));
	std::cout << "\ttail_variate    = "; output_range(boost::accumulators::tail_variate(acc));
	std::cout << std::endl;

	acc(42.1, boost::accumulators::covariate1 = 4.21);
	acc(41.1, boost::accumulators::covariate1 = 4.11);
	acc(42.1, boost::accumulators::covariate1 = 4.21);
	acc(41.1, boost::accumulators::covariate1 = 4.11);

	std::cout << "\ttail            = "; output_range(boost::accumulators::tail(acc));
	std::cout << "\ttail_variate    = "; output_range(boost::accumulators::tail_variate(acc));
	std::cout << std::endl;

	acc(32.1, boost::accumulators::covariate1 = 3.21);
	acc(31.1, boost::accumulators::covariate1 = 3.11);
	acc(32.1, boost::accumulators::covariate1 = 3.21);
	acc(31.1, boost::accumulators::covariate1 = 3.11);

	std::cout << "\ttail            = "; output_range(boost::accumulators::tail(acc));
	std::cout << "\ttail_variate    = "; output_range(boost::accumulators::tail_variate(acc));
}

// REF [file] >> ${BOOST_HOME}/libs/accumulators/example/main.cpp
void simple_example_3()
{
	// weight == double.
	double w = 1.0;

	// Simple weighted calculation.
	{
		// stats that depend on the weight are made external.
		boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::mean>, double> acc;

		acc(0., boost::accumulators::weight = w);
		acc(1., boost::accumulators::weight = w);
		acc(-1., boost::accumulators::weight = w);
		acc(3.14159, boost::accumulators::weight = w);

		std::cout << "\tmean(acc)       = " << boost::accumulators::mean(acc) << std::endl;
	}

	// Weighted calculation with an external weight accumulator.
	{
		// stats that depend on the weight are made external.
		boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::mean>, boost::accumulators::external<double> > acc;

		// Here's an external weight accumulator.
		boost::accumulators::accumulator_set<void, boost::accumulators::stats<boost::accumulators::tag::sum_of_weights>, double> weight_acc;

		weight_acc(boost::accumulators::weight = w); acc(0., boost::accumulators::weight = w);
		weight_acc(boost::accumulators::weight = w); acc(1., boost::accumulators::weight = w);
		weight_acc(boost::accumulators::weight = w); acc(-1., boost::accumulators::weight = w);
		weight_acc(boost::accumulators::weight = w); acc(3.14159, boost::accumulators::weight = w);

		std::cout << "\tmean(acc)       = " << boost::accumulators::mean(acc, boost::accumulators::weights = weight_acc) << std::endl;
	}
}

void basic()
{
	{
		boost::accumulators::accumulator_set<double, boost::accumulators::features<boost::accumulators::tag::mean, boost::accumulators::tag::variance>, int> acc;
		//boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::mean, boost::accumulators::tag::variance>, int> acc;
		// { 8 9 10 10 10 10 11 12 }.
		acc(8, boost::accumulators::weight = 1);
		acc(9, boost::accumulators::weight = 1);
		acc(10, boost::accumulators::weight = 4);
		acc(11, boost::accumulators::weight = 1);
		acc(12, boost::accumulators::weight = 1);
		std::cout << "\tmean(acc)     = " << boost::accumulators::mean(acc) << std::endl;
		std::cout << "\tvariance(acc) = " << boost::accumulators::variance(acc) << std::endl;
	}

	{
		boost::accumulators::accumulator_set<double, boost::accumulators::features<boost::accumulators::tag::weighted_mean, boost::accumulators::tag::weighted_moment<2> >, int> acc;
		//boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::weighted_mean, boost::accumulators::tag::weighted_moment<2> >, int> acc;
		// { 8 9 10 10 10 10 11 12 }.
		acc(8, boost::accumulators::weight = 1);
		acc(9, boost::accumulators::weight = 1);
		acc(10, boost::accumulators::weight = 4);
		acc(11, boost::accumulators::weight = 1);
		acc(12, boost::accumulators::weight = 1);
		std::cout << "\tweighted mean(acc)   = " << boost::accumulators::weighted_mean(acc) << std::endl;
		std::cout << "\tweighted moment(acc) = " << boost::accumulators::weighted_moment<2>(acc) << std::endl;  // A raw moment or crude moment.
	}
}

}  // namespace local
}  // unnamed namespace

void accumulator()
{
	local::simple_example_1();
	local::simple_example_2();
	local::simple_example_3();

	local::basic();
}
