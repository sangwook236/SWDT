#include <iostream>
#include <stdexcept>


namespace {
namespace local {

void pictorial_structure_revisited(int argc, char **argv)
{
	int pictorial_structure_revisited_partapp_main(int argc, char *argv[]);

	pictorial_structure_revisited_partapp_main(argc, argv);
}

}  // namespace local
}  // unnamed namespace

void pictorial_structure(int argc, char **argv)
{
	local::pictorial_structure_revisited(argc, argv);
}
