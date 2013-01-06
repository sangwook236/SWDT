#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <algorithm>
#include <string>
#include <iostream>


namespace {
namespace local {

static const size_t N = 23;

class SubStringFinder
{
public:
	SubStringFinder(std::string &s, size_t *m, size_t *p)
	: str(s), max_array(m), pos_array(p)
	{}

public:
	void operator()(const tbb::blocked_range<std::size_t> &r) const
	{
		for (std::size_t i = r.begin(); i != r.end(); ++i)
		{
			std::size_t max_size = 0, max_pos = 0;
			for (std::size_t j = 0; j < str.size(); ++j)
				if (j != i)
				{
					std::size_t limit = str.size() - std::max(i,j);
					for (std::size_t k = 0; k < limit; ++k)
					{
						if (str[i + k] != str[j + k]) break;
						if (k > max_size)
						{
							max_size = k;
							max_pos = j;
						}
					}
				}

			max_array[i] = max_size;
			pos_array[i] = max_pos;
		}
	}

private:
	const std::string str;
	size_t *max_array;
	size_t *pos_array;
};

void simple_loop_parallelization_1()
{
	std::string str[N] = { std::string("a"), std::string("b") };
	for (std::size_t i = 2; i < N; ++i) str[i] = str[i-1] + str[i-2];
	std::string &to_scan = str[N - 1];
	const std::size_t num_elem = to_scan.size();
	
	std::size_t *max = new size_t [num_elem];
	std::size_t *pos = new size_t [num_elem];
	
	//tbb::parallel_for(tbb::blocked_range<std::size_t>(0, num_elem, 1), SubStringFinder(to_scan, max, pos), tbb::auto_partitioner());
	tbb::parallel_for(tbb::blocked_range<std::size_t>(0, num_elem), SubStringFinder(to_scan, max, pos));
	
	for (std::size_t i = 0; i < num_elem; ++i)
		std::cout << " " << max[i] << "(" << pos[i] << ")" << std::endl;

	delete [] pos;
	delete [] max;
}

}  // namespace local
}  // unnamed namespace

namespace tbb {

void simple_loop_parallelization()
{
	local::simple_loop_parallelization_1();
}

}  // namespace tbb
