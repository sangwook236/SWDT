#include <gmock/gmock.h>
#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_gmock {

}  // namespace my_gmock

// REF [site] >> https://code.google.com/p/googlemock/wiki/ForDummies
int gmock_main(int argc, char *argv[])
{
#if 1
    // The following line must be executed to initialize Google Mock (and Google Test) before running the tests.
    testing::InitGoogleMock(&argc, argv);
    //testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
#else
    // If you want to use something other than Google Test (e.g. CppUnit or CxxTest) as your testing framework.

    // The following line causes Google Mock to throw an exception on failure, which will be interpreted by your testing framework as a test failure.
    testing::GTEST_FLAG(throw_on_failure) = true;
    testing::InitGoogleMock(&argc, argv);

    // whatever your testing framework requires ...
#endif
}
