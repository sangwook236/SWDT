#include <gtest/gtest.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_gtest {

}  // namespace my_gtest

int gtest_main(int argc, char *argv[])
{
	// Usage:
	//	REF [site] >> https://google.github.io/googletest/
	//
	//	Command line:
	//		my_gtest --gtest_output="xml:/path/to/gtest_report.xml" --gtest_repeat=2 --gtest_break_on_failure --gtest_filter=<test string>
	//			The format for <test string> is a series of wildcard patterns separated by colons (:).
	//				--gtest_filter=* : runs all tests
	//				--gtest_filter=SquareRoot* : runs tests which start with SquareRoot.
	//				--gtest_filter=SquareRootTest.*-SquareRootTest.Zero*
	//					SquareRootTest.* means all tests belonging to SquareRootTest.
	//					-SquareRootTest.Zero* means don't run those tests whose names begin with Zero.
	//		e.g.)
	//			my_gtest --help
	//			my_gtest --gtest_list_tests
	//			my_gtest ./s --number-of-input=5 --gtest_output="xml:/path/to/gtest_report.xml" --gtest_filter="Test_Cases1*"
	//
	//	Source code:
	//		GTEST_FLAG_SET(output, "xml:/path/to/gtest_report.xml");
	//		GTEST_FLAG_SET(filter, "Test_Cases1*");
	//		GTEST_FLAG_SET(repeat, 2);
	//
	//	Environment variables:
	//		export GTEST_OUTPUT="xml:/path/to/gtest_report.xml"
	//		export GTEST_FILTER="Test_Cases1*"
	//		export GTEST_REPEAT=2

#if 0
	//testing::GTEST_FLAG(output) = "xml:/path/to/gtest_report.xml";
	//testing::GTEST_FLAG(filter) = "Test_Cases1*";
	//testing::GTEST_FLAG(repeat) = 2;
	//testing::GTEST_FLAG(death_test_style) = "threadsafe";
	//testing::GTEST_FLAG(throw_on_failure) = true;
#else
	//GTEST_FLAG_SET(output, "xml:/path/to/gtest_report.xml");
	//GTEST_FLAG_SET(filter, "Test_Cases1*");
	//GTEST_FLAG_SET(repeat, 2);
	//GTEST_FLAG_SET(death_test_style, "threadsafe");
	//GTEST_FLAG_SET(throw_on_failure, true);
#endif

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
