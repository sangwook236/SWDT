#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_region_growing {

void region_growing_by_contrast();
void andres_seeded_region_growing();

}  // namespace my_region_growing

int region_growing_main(int argc, char *argv[])
{
	// seeded regin growing (SRG).
	//	-. [ref] "Seeded Region Growing", R. Adams and L. Bischof, TPAMI, 1994.
	//	-. [ref] vigra library.
	//		${CPP_RND_HOME}/src/machine_vision/vigra.
	//		${VIGRA_HOME}/include/vigra/seededregiongrowing.hxx & seededregiongrowing3d.hxx.

	// regin growing: a new approach.
	//	-. [ref] "Region Growing: A New Approach", S. A. Hojjatoleslami and J. Kittler, TIP, 1998.
	my_region_growing::region_growing_by_contrast();

	// Bjoern Andres's seeded region growing in n-dimensional grid graphs, in linear time.
	//my_region_growing::andres_seeded_region_growing();

	return 0;
}
