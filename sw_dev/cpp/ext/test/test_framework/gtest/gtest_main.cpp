#include <gtest/gtest.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_gtest {

}  // namespace my_gtest

int gtest_main(int argc, char *argv[])
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
