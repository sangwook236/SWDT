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

void f()
{
    int *x = (int *)malloc(10 * sizeof(int));

    // problem 1: heap block overrun
    x[10] = 0;

    // problem 2: memory leak - x not freed
}

int main(int argc, char* argv[])
{

	try
	{
	    f();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::flush;
	std::cin.get();

    return 0;
}

