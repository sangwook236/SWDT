#include "../particle_filter_object_tracking_lib/defs.h"
#include "../particle_filter_object_tracking_lib/utils.h"
#include "../particle_filter_object_tracking_lib/particles.h"
#include "../particle_filter_object_tracking_lib/observation.h"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdio>
#include <ctime>


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
