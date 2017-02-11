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
	// Seeded regin growing (SRG).
	//	- REF [paper] >> "Seeded Region Growing", R. Adams and L. Bischof, TPAMI 1994.
	//	- REF [library] >> Vigra library.
	//		${CPP_RND_HOME}/src/machine_vision/vigra.
	//		${VIGRA_HOME}/include/vigra/seededregiongrowing.hxx & seededregiongrowing3d.hxx.

	// Bjoern Andres's seeded region growing in n-dimensional grid graphs, in linear time.
	//my_region_growing::andres_seeded_region_growing();

	// Regin growing: a new approach.
	//	- REF [paper] >> "Region Growing: A New Approach", S. A. Hojjatoleslami and J. Kittler, TIP 1998.
	my_region_growing::region_growing_by_contrast();

	return 0;
}
