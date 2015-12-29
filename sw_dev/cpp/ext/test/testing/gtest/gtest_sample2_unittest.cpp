#include "gtest_sample2.h"
#include <gtest/gtest.h>

// In this example, we test the MyString class (a simple string).

// Tests the default c'tor.
TEST(MyString, DefaultConstructor)
{
	const MyString s;

	EXPECT_STREQ(NULL, s.c_string());

	EXPECT_EQ(0u, s.Length());
}

const char kHelloString[] = "Hello, world!";

// Tests the c'tor that accepts a C string.
TEST(MyString, ConstructorFromCString)
{
	const MyString s(kHelloString);
	EXPECT_TRUE(strcmp(s.c_string(), kHelloString) == 0);
	EXPECT_EQ(sizeof(kHelloString) / sizeof(kHelloString[0]) - 1, s.Length());
}

// Tests the copy c'tor.
TEST(MyString, CopyConstructor)
{
	const MyString s1(kHelloString);
	const MyString s2 = s1;
	EXPECT_TRUE(strcmp(s2.c_string(), kHelloString) == 0);
}

// Tests the Set method.
TEST(MyString, Set)
{
	MyString s;

	s.Set(kHelloString);
	EXPECT_TRUE(strcmp(s.c_string(), kHelloString) == 0);

	// Set should work when the input pointer is the same as the one
	// already in the MyString object.
	s.Set(s.c_string());
	EXPECT_TRUE(strcmp(s.c_string(), kHelloString) == 0);

	// Can we set the MyString to NULL?
	s.Set(NULL);
	EXPECT_STREQ(NULL, s.c_string());
}
