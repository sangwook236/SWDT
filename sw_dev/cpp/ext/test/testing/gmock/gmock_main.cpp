#include <stdexcept>
#include <gmock/gmock.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_gmock {

}  // namespace my_gmock

int gmock_main(int argc, char *argv[])
{
	// REF [site] >>
	//	https://google.github.io/googletest/gmock_for_dummies.html
	//	https://google.github.io/googletest/

#if 1
	// The following line must be executed to initialize Google Mock (and Google Test) before running the tests.
	testing::InitGoogleMock(&argc, argv);
	//testing::InitGoogleTest(&argc, argv);  // Do not need to be called.
	return RUN_ALL_TESTS();
#else
	// If you want to use something other than Google Test (e.g. CppUnit or CxxTest) as your testing framework.

	// The following line causes Google Mock to throw an exception on failure, which will be interpreted by your testing framework as a test failure.
	testing::GTEST_FLAG(throw_on_failure) = true;
	testing::InitGoogleMock(&argc, argv);

	// Whatever your testing framework requires ...
#endif
}
