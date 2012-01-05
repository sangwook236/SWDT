//#include <vld/vld.h>   // don't need to include here. the header file is included in main.cpp
#include <iostream>


namespace {
namespace local {

void malloc_free(const bool leakage)
{
	const size_t BLOCK = 100;
	int *p = (int *)malloc(BLOCK * sizeof(int));

	for (size_t i = 0; i < BLOCK; ++i)
		p[i] = i;

	if (!leakage)
	{
		free(p);
		p = NULL;
	}
}

void new_delete(const bool leakage)
{
	//
	int *p1 = new int;
	
	*p1 = 100;

	if (!leakage)
	{
		delete p1;
		p1 = NULL;
	}

	//
	const size_t BLOCK = 100;
	int *p2 = new int [BLOCK];

	for (size_t i = 0; i < BLOCK; ++i)
		p2[i] = i;

	if (!leakage)
	{
		delete [] p2;
		p2 = NULL;
	}
}

}  // namespace local
}  // unnamed namespace

void basic(const bool leakage)
{
	local::malloc_free(leakage);
	local::new_delete(leakage);
}
