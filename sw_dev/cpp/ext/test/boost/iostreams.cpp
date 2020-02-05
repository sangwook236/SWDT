#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/stream.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <string>
#include <chrono>

namespace {
namespace local {

void memory_mapped_file_sink_example()
{
    const std::vector<std::string> strs(2000000, "A1");

    //--------------------
    {
        const std::string mmap_filepath("./mmap_file.txt");

        boost::iostreams::mapped_file_params params;
        params.path = mmap_filepath;
        params.new_file_size = 30ul << 30;
        params.flags = boost::iostreams::mapped_file::mapmode::readwrite;

        const auto start_time(std::chrono::high_resolution_clock::now());
        boost::iostreams::stream<boost::iostreams::mapped_file_sink> out(params);
        std::copy(strs.begin(), strs.end(), std::ostream_iterator<std::string>(out, "\n"));
        std::cout << "Elapsed time (memory mapped file) = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << std::endl;
    }

    //--------------------
    {
        const std::string plain_filepath("./plain_file.txt");

        const auto start_time(std::chrono::high_resolution_clock::now());
        std::ofstream stream(plain_filepath, std::ios::out);
        std::copy(strs.begin(), strs.end(), std::ostream_iterator<std::string>(stream, "\n"));
        std::cout << "Elapsed time (plain file) = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << std::endl;
    }
}

void memory_mapped_file_source_example()
{
    const std::string mmap_filepath("./mmap_file.txt");
    const int num_elements = 1000000;
    const int num_bytes = num_elements * sizeof(int);

    boost::iostreams::mapped_file_source file(mmap_filepath, num_bytes);

    // Check if file was successfully opened.
    if (file.is_open())
    {
        // Get pointer to the data.
        const char *data = (char*)file.data();

        // Do something with the data.
        const int num_data = std::min(num_elements, 100);
        std::copy(data, data + num_data, std::ostream_iterator<char>(std::cout, "\n"));
    }
    else
    {
        std::cout << "Could not map the file, " << mmap_filepath << std::endl;
    }
}

}  // namespace local
}  // unnamed namespace

void iostreams()
{
	local::memory_mapped_file_sink_example();
    //local::memory_mapped_file_source_example();
}
