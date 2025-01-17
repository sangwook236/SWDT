#include <vector>
#include <string>
#include <iterator>
#include <iostream>


namespace {
namespace local {

#if defined(_MSC_VER)
__declspec(deprecated("Use Func() instead.")) void deprecated_func()
#elif defined(__GNUC__) || defined(__clang__)
__attribute__((deprecated("Use Func() instead."))) void deprecated_func()
#else
[[deprecated("Use Func() instead.")]] void deprecated_func()
#endif
{
	std::cout << "A deprecated function, func() is called." << std::endl;
}

}  // namespace local
}  // unnamed namespace

void deprecated_test()
{
	[[deprecated]] int i = 100;
	std::cout << "i = " << i << std::endl;

#if defined(_MSC_VER)
	__declspec(deprecated) int j = 200;
#elif defined(__GNUC__) || defined(__clang__)
	__attribute__((deprecated)) int j = 200;
#else
	[[deprecated]] int j = 200;
#endif
	std::cout << "j = " << j << std::endl;

	local::deprecated_func();
}
