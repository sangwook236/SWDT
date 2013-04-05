#if defined(WIN32) || defined(_WIN32)
#include <stdexcept>
#else
#include "../movmf_lib/movmf.h"
#endif
#include <iostream>


namespace {
namespace local {

#if defined(WIN32) || defined(_WIN32)
#else
// ${MoVMF_HOME}/main.cc
void basic_example()
{
	// [ref] http://people.kyb.tuebingen.mpg.de/suvrit/work/progs/movmf/vmfREADME.html
/*
	USAGE: moVMF [switches] word-doc-file
		-a algorithm
		   s: moVMF algorithm (default)
		-i [s|p|r]
		  initialization method:
			 s -- subsampling
			 p -- random perturbation (default)
			 r -- totally random
			 f -- read from file
		-c number-of-clusters
		-e epsilon
		-s suppress output
		-v version-number
		-n no dump
		-d dump the clustering process
		-p perturbation-magnitude the distance between initial concept vectors
			and the centroid will be less than this.
		-N number-of-samples
		-O the name of output matrix
		-t scaling scheme
		-K lower bound on Kappa
		-Z upper bound on Kappa
		-S Run SOFT moVMF (default is HARD moVMF)
		-E encoding-scheme
		   1: normalized term frequency (default)
		   2: normalized term frequency inverse document frequency
		-o objective-function
			1: nonweighted (default)
			2: weighted

	sample commandline:
		Suppose we have a dataset called 'data'.
		Thus we should have the following files in the current directory: data_dim, data_txx_nz, data_col_ccs, data_row_ccs.
			moVMF -c 3 -S -K 1 -Z 100 -t txx -O clusters data
		The clusters will be left in a file of the name clusters_txx_doctoclus.3.
*/

	const std::string data_filename("data");
	const int my_argc = 13;
	const char *my_argv[] = { "moVMF", "-c", "3", "-S", "-K", "1", "-Z", "100", "-t", "txx", "-O", "clusters", data_filename.c_str() };

	movmf *movmf_engine = new movmf(my_argc, (char **)my_argv);
	const int retval = movmf_engine->run();
}
#endif

}  // namespace local
}  // unnamed namespace

namespace my_movmf {

}  // namespace my_movmf

// MoVMF library supports C++ & Matlab.
// MoVMF library was designed for the purpose of using command line as an executable file, but not a library.

int movmf_main(int argc, char *argv[])
{
#if defined(WIN32) || defined(_WIN32)
	throw std::runtime_error("not yet implemented in Windows");
#else
	local::basic_example();
#endif

	return 0;
}
