#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_particle_filter_object_tracking {

void observe_example();
void track_example();

}  // namespace my_particle_filter_object_tracking

int particle_filter_object_tracking_main(int argc, char *argv[])
{
	//my_particle_filter_object_tracking::observe_example();
	my_particle_filter_object_tracking::track_example();

	return 0;
}
