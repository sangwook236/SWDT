#include <iostream>
#include <cstdlib>


//-----------------------------------------------------
// [usage]
// 1. if you normally run your program
//  myprog arg1 arg2
// 2. if you invoke Valgrind
//  valgrind --tool=memcheck myprog arg1 arg2
//  Memcheck is the default, so if you want to use it you can omit the --tool option.
// 3. if you invoke Valgrind
//  valgrind --leak-check=yes myprog arg1 arg2
//  The --leak-check option turns on the detailed memory leak detector

//-----------------------------------------------------
// Is there a good Valgrind substitute for Windows?
//	[ref] http://stackoverflow.com/questions/413477/is-there-a-good-valgrind-substitute-for-windows

namespace {
namespace local {

void f()
{
    int *x = (int *)malloc(10 * sizeof(int));

    // problem 1: heap block overrun
    x[10] = 0;

    // problem 2: memory leak - x not freed
}

}  // namespace local
}  // unnamed namespace

namespace valgrind {

}  // namespace valgrind

int valgrind_main(int argc, char* argv[])
{
	local::f();

    return 0;
}

