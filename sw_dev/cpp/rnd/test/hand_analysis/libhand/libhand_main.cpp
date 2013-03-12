#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_libhand {

void file_dialog_example();
void render_hog_descriptor_example();
void bend_one_finger_example();

}  // namespace my_libhand

int libhand_main(int argc, char *argv[])
{
	//my_libhand::file_dialog_example();
	//my_libhand::render_hog_descriptor_example();
	my_libhand::bend_one_finger_example();

	return 0;
}
