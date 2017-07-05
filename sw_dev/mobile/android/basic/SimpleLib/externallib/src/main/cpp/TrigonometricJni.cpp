#include "TrigonometricJni.h"
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jdouble JNICALL Java_com_sangwook_simplejni_TrigonometricJni_sin(JNIEnv *env, jobject obj, jdouble val)
{
    return std::sin(val);
}

JNIEXPORT jdouble JNICALL Java_com_sangwook_simplejni_TrigonometricJni_cos(JNIEnv *env, jobject obj, jdouble val)
{
    return std::cos(val);
}

#ifdef __cplusplus
}
#endif
