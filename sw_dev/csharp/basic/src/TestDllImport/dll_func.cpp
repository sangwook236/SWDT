#include "dll_func.h"
#include <string>
#include <iostream>


#if defined(__cplusplus)
extern "C" {
#endif

int func_in_dll(int i, char *str, struct_in_dll *data)
{
	std::cout << "func_in_dll called" << std::endl;
	std::cout << "---------------" << std::endl;
	std::cout << "  an int value = " << i << std::endl;

	strcpy(str, "String filled in DLL");

	std::cout << "  data.count = " << data->count_ << std::endl;;
	for (int k = 0; k < data->count_; ++k)
	{
		std::cout << "  " << data->data_[k];
	}
	std::cout << std::endl;

	std::cout << "returning from func_in_dll" << std::endl;
	std::cout << "-----------------------" << std::endl;

	return 2 * i;
}

#if defined(__cplusplus)
}
#endif
