#include <vl/aib.h>
#include <iostream>

namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_vlfeat {

// agglomerative information bottleneck (AIB)
void aib()
{
	const vl_uint32 nrows = 10;
	const vl_uint32 ncols = 3;
	double Pic[] = {
		0.6813,    0.3028,    0.8216,
		0.3795,    0.5417,    0.6449,
		0.8318,    0.1509,    0.8180,
		0.5028,    0.6979,    0.6602,
		0.7095,    0.3784,    0.3420,
		0.4289,    0.8600,    0.2897,
		0.3046,    0.8537,    0.3412,
		0.1897,    0.5936,    0.5341,
		0.1934,    0.4966,    0.7271,
		0.6822,    0.8998,    0.3093,
	};

	std::cout << "Pic = [";
	for (vl_uint32 r = 0; r < nrows; ++r)
	{
		for (vl_uint32 c = 0; c < ncols; ++c)
			std::cout << Pic[r*ncols + c] << ' ';
		std::cout << "; ..." << std::endl;
	}
	std::cout << "];" << std::endl;

	//
	std::cout << "AIB starting" << std::endl;
	{
		VlAIB *aib = vl_aib_new(Pic, nrows, ncols);
		vl_aib_process(aib);

		// parents always has size 2 * nrows - 1
		vl_uint *parents = vl_aib_get_parents(aib);
		for (vl_uint32 r = 0; r < 2 * nrows - 1; ++r)
			std::cout << r << " => " << parents[r] << std::endl;

		vl_aib_delete(aib);
	}

	std::cout << "AIB done" << std::endl;
}

}  // namespace my_vlfeat
