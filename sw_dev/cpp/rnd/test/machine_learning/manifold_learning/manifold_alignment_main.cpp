//include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_manifold_alignment {

void locally_linear_coordination();

}  // namespace my_manifold_alignment

int manifold_alignment_main(int argc, char *argv[])
{
    std::cout << "\nAutomatic Alignment of Local Representations ------------------------" << std::endl;
    my_manifold_alignment::locally_linear_coordination();

	return 0;
}
