#include <vector>
#include <string>
#include <iostream>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_pipeline.h>


namespace {
namespace local {

void simple_test()
{
	// tbb::parallel_for
	{
		const size_t n = 1000000;
		std::vector<int> data(n);

		// Initialize array with sequential values
		for (size_t i = 0; i < n; ++i)
		{
			data[i] = int(i);
		}

		// Parallel computation: square each element
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, n),
			[&](const tbb::blocked_range<size_t>& range) {
				for (size_t i = range.begin(); i != range.end(); ++i)
				{
					data[i] = data[i] * data[i];
				}
			}
		);

		// Verify first few results
		std::cout << "First 10 squared values:" << std::endl;
		for (size_t i = 0; i < 10; ++i)
		{
			std::cout << i << "^2 = " << data[i] << std::endl;
		}
	}

	// tbb::parallel_for_each
	{
		std::vector<std::string> words{ "apple", "banana", "cherry", "date", "elderberry" };

		// Parallel operation: convert each word to uppercase
		tbb::parallel_for_each(
			words.begin(), words.end(),
			[](std::string& word) {
				for (char& c : word)
				{
					c = (char)std::toupper(int(c));
				}
			}
		);

		// Output the results
		std::cout << "Uppercase words: ";
		for (const auto& word : words)
			std::cout << word << ", ";
		std::cout << std::endl;
	}

	{
		std::vector<int> data{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

		// Parallel computation: square each element
		tbb::parallel_for_each(data, [](int& value) {
			value = value * value;
		});

		// Print results
		for (int val : data)
			std::cout << val << " ";
		std::cout << std::endl;
	}

	// tbb::parallel_reduce
	{
		const size_t n = 1000000;
		std::vector<int> data(n);

		// Initialize array with sequential values
		for (size_t i = 0; i < n; ++i)
		{
			data[i] = int(i + 1);  // Avoid zero to prevent multiplication by zero
		}

		// Parallel computation: compute the product of all elements
		long long product = tbb::parallel_reduce(
			tbb::blocked_range<size_t>(0, n),
			1LL,  // Identity value for multiplication
			[&](const tbb::blocked_range<size_t>& range, long long init) -> long long {
				for (size_t i = range.begin(); i != range.end(); ++i)
				{
					init *= data[i];
				}
				return init;
			},
			std::multiplies<long long>()  // Combine results from different ranges
		);

		std::cout << "Product of first " << n << " natural numbers = " << product << std::endl;
	}

	{
		std::vector<int> data{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

		// Parallel sum reduction
		const int sum = tbb::parallel_reduce(
			tbb::blocked_range<size_t>(0, data.size()),
			0,  // Identity value
			[&](const tbb::blocked_range<size_t>& range, int init) -> int {
				for (size_t i = range.begin(); i != range.end(); ++i)
				{
					init += data[i];
				}
				return init;
			},
			[](int a, int b) -> int {
				return a + b;  // Reduction operation
			}
		);

		std::cout << "Sum: " << sum << std::endl;
	}

	// tbb::parallel_pipeline
	{
		const int num_items = 10;
	
		tbb::parallel_pipeline(
			8,  // Max tokens in pipeline
			tbb::make_filter<void, int>(
				tbb::filter_mode::serial_in_order,
				[&](tbb::flow_control& fc) -> int {
					static int counter = 0;
					if (counter >= num_items)
					{
						fc.stop();
						return 0;
					}
					return ++counter;
				}
			) &
			tbb::make_filter<int, int>(
				tbb::filter_mode::parallel,
				[](int value) -> int {
					return value * value;  // Square the value
				}
			) &
			tbb::make_filter<int, void>(
				tbb::filter_mode::serial_in_order,
				[](int value) {
					std::cout << value << " ";
				}
			)
		);

		std::cout << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_tbb {

}  // namespace my_tbb

int tbb_main(int argc, char *argv[])
{
	local::simple_test();

	return 0;
}

