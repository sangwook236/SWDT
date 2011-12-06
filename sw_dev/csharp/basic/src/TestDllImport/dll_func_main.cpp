#include "dll_func.h"
#include <iostream>


int main()
{
	char str[21];
	struct struct_in_dll S;

	S.count_ = 10;
	S.data_ = new int [10];

	for (int i = 0; i < 10; ++i)
		S.data_[i] = i;

	func_in_dll(42, str, &S);

	std::cout << "str: " << str << std::endl;

	delete [] S.data_;

	return 0;
}
