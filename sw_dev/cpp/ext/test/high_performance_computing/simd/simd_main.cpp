#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_simd {

void sse();

}  // namespace my_simd

int simd_main(int argc, char *argv[])
{
	my_simd::sse();

    return 0;
}

