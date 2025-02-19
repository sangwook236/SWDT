#include <chrono>
#include <iterator>
#include <vector>
#include <algorithm>
#include <execution>
#include <iostream>


using namespace std::literals::chrono_literals;

namespace {
namespace local {

void map_reduce_test()
{
	// Goal: calculate the sum of the squares of odd numbers between 1 and 1000

	const size_t MAX_VAL = 1'000;
	const size_t MAX_ITERATIONS = 100'000;

	double total;
	std::chrono::system_clock::time_point start_time;

#if 0
	std::vector<int> v(MAX_VAL);
	std::generate(v.begin(), v.end(), [n = 1]() mutable { return n++; });
	//std::iota(v.begin(), v.end(), 1);
#else
	std::vector<int> v;
	v.reserve(MAX_VAL);
	std::generate_n(std::back_inserter(v), MAX_VAL, [n = 1]() mutable { return n++; });
#endif

	//-----
	{
		start_time = std::chrono::high_resolution_clock::now();
		total = 0;
		for (size_t iter = 0; iter < MAX_ITERATIONS; ++iter)
		{
			int sum = 0;
			for (size_t i = 0; i < v.size(); ++i)
				if (v[i] % 2 == 1)
					sum += v[i] * v[i];
			total += sum;
		}
		std::cout << "For loop (1) = " << int(total / MAX_ITERATIONS) << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
	}

	{
		start_time = std::chrono::high_resolution_clock::now();
		total = 0;
		for (size_t iter = 0; iter < MAX_ITERATIONS; ++iter)
		{
			int sum = 0;
			for (const auto &val: v)
				if (val % 2 == 1)
					sum += val * val;
			total += sum;
		}
		std::cout << "For loop (2) = " << int(total / MAX_ITERATIONS) << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
	}

	{
		start_time = std::chrono::high_resolution_clock::now();
		total = 0;
		for (size_t iter = 0; iter < MAX_ITERATIONS; ++iter)
		{
			int sum = 0;
			std::for_each(v.begin(), v.end(), [&sum](auto val) { if (val % 2 == 1) sum += val * val; });
			total += sum;
		}
		std::cout << "for_each() = " << int(total / MAX_ITERATIONS) << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
	}

	//-----
	{
		// Filter-map-reduce as separate steps

		start_time = std::chrono::high_resolution_clock::now();
		total = 0;
		for (size_t iter = 0; iter < MAX_ITERATIONS; ++iter)
		{
			// Filter
			std::vector<int> v_f;
			v_f.reserve(v.size());
			std::copy_if(v.begin(), v.end(), std::back_inserter(v_f), [](auto val) -> auto { return val % 2 == 1; });

			// Map
			std::vector<int> v_m;
			v_m.reserve(v.size());
			std::transform(v_f.cbegin(), v_f.cend(), std::back_inserter(v_m), [](auto val) -> auto { return val * val; });
			//std::transform(std::execution::seq, v_f.cbegin(), v_f.cend(), std::back_inserter(v_m), [](auto val) -> auto { return val * val; });
			//std::transform(std::execution::par, v_f.cbegin(), v_f.cend(), std::back_inserter(v_m), [](auto val) -> auto { return val * val; });

			// Reduce
#if 0
			const auto sum = std::accumulate(v_m.cbegin(), v_m.cend(), 0);
#else
			//const auto sum = std::reduce(v_m.cbegin(), v_m.cend(), 0);
			//const auto sum = std::reduce(std::execution::seq, v_m.cbegin(), v_m.cend(), 0);
			const auto sum = std::reduce(std::execution::par, v_m.cbegin(), v_m.cend(), 0);
#endif

			total += sum;
		}
		std::cout << "Filter-map-reduce = " << int(total / MAX_ITERATIONS) << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
	}

	//-----
	{
		// Filter-map-reduce as a single step

		//auto squared_sum_functor = [](auto sum, auto val) -> auto { return sum + val * val; };
		auto squared_sum_functor = [](auto sum, auto val) -> auto { return val % 2 == 1 ? (sum + val * val) : sum; };

		start_time = std::chrono::high_resolution_clock::now();
		total = 0;
		for (size_t iter = 0; iter < MAX_ITERATIONS; ++iter)
		{
			const auto sum_a = std::accumulate(v.cbegin(), v.cend(), 0, squared_sum_functor);
			total += sum_a;
		}
		std::cout << "accumulate() = " << int(total / MAX_ITERATIONS) << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;

		start_time = std::chrono::high_resolution_clock::now();
		total = 0;
		for (size_t iter = 0; iter < MAX_ITERATIONS; ++iter)
		{
			// FIXME [fix] >> incorrect result
			//const auto sum_r = std::reduce(v.cbegin(), v.cend(), 0, squared_sum_functor);
			//const auto sum_r = std::reduce(std::execution::seq, v.cbegin(), v.cend(), 0, squared_sum_functor);
			const auto sum_r = std::reduce(std::execution::par, v.cbegin(), v.cend(), 0, squared_sum_functor);
			total += sum_r;
		}
		std::cout << "reduce() = " << int(total / MAX_ITERATIONS) << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;

		start_time = std::chrono::high_resolution_clock::now();
		total = 0;
		for (size_t iter = 0; iter < MAX_ITERATIONS; ++iter)
		{
			//const auto sum_tr = std::transform_reduce(v.cbegin(), v.cend(), 0, std::plus{}, [](auto val) -> auto { return val % 2 == 1 ? (val * val) : 0; });
			//const auto sum_tr = std::transform_reduce(std::execution::seq, v.cbegin(), v.cend(), 0, std::plus{}, [](auto val) -> auto { return val % 2 == 1 ? (val * val) : 0; });
			const auto sum_tr = std::transform_reduce(std::execution::par, v.cbegin(), v.cend(), 0, std::plus{}, [](auto val) -> auto { return val % 2 == 1 ? (val * val) : 0; });
			total += sum_tr;
		}
		std::cout << "transform_reduce() = " << int(total / MAX_ITERATIONS) << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void algorithm()
{
	local::map_reduce_test();
}
