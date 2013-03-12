#include <libhand/file_dialog.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_libhand {

// [ref] ${LIBHAND_HOME}/examples/file_dialog_test.cpp
void file_dialog_example()
{
    libhand::FileDialog f;

    // The simplest open dialog invocation
    std::cout << "File name: " << f.Open() << std::endl;

    // This shows off the main features of the file open/save dialog.
    f.SetTitle("Open a YAML file");

    f.AddExtension(libhand::FileExtension("YAML file", 2, ".yml", ".yaml"));
    f.AddExtension(libhand::FileExtension("All files", 1, "*"));

    f.SetDefaultExtension(".yml");
    f.SetDefaultName("scene_spec.yml");

    std::cout << "File name: " << f.Open() << std::endl;

    // This shows off a save dialog. SetDefaultFile() can be used to suggest a particular file in a particular directory.
    f.SetTitle("Save a YAML file");
    f.SetDefaultFile("./hand_analysis_data/libhand/my_new_spec.yml");

    std::cout << "Save file name: " << f.Save() << std::endl;
}

}  // namespace my_libhand
