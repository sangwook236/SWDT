#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_scip {

void bin_packing_example(int argc, char *argv[]);

}  // namespace my_scip

int scip_main(int argc, char *argv[])
{
	my_scip::bin_packing_example(argc, argv);

    return 0;
}
