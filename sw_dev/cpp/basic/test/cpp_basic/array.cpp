#include <iostream>


namespace {
namespace local {
	
}  // namespace local
}  // unnamed namespace

void test_array()
{
	const char char_arr[] = { 0x00, 0x01, 0x02, 0x03 };

	const size_t arrCount = sizeof(char_arr) / sizeof(char_arr[0]);

	std::wcout << L"size of array: " << arrCount << std::endl;
}
