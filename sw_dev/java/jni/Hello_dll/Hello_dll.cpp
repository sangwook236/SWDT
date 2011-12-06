#include "../HelloJniClass.h"
#include <jni.h>
#include <iostream>


#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_HelloJniClass_Hello(JNIEnv *, jobject)
{
	std::cout << "Hello, JNI !!!" << std::endl;
}

#ifdef __cplusplus
}
#endif
