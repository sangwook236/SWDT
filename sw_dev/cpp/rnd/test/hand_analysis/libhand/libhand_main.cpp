#include <iostream>
#if defined(WIN32) || defined(_WIN32)
#include <stdexcept>
#endif

namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_libhand {

#if defined(WIN32) || defined(_WIN32)
#else
void file_dialog_example();
void render_hog_descriptor_example();
void bend_one_finger_example();
#endif

}  // namespace my_libhand

int libhand_main(int argc, char *argv[])
{
#if defined(WIN32) || defined(_WIN32)
	throw std::runtime_error("not yet implemented in Windows");
#else
	//my_libhand::file_dialog_example();
	//my_libhand::render_hog_descriptor_example();
	my_libhand::bend_one_finger_example();
#endif

	return 0;
}
