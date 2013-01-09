#include <cuda_runtime.h>
#include <iostream>


#if defined(__cplusplus)
extern "C" {
#endif
	
void HelloWorld()
{
	std::cout << "Hello World!" << std::endl;
}

#if defined(__cplusplus)
}  // extern "C"
#endif
