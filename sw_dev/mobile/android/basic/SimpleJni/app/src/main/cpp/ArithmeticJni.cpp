//
// Created by sangwook on 7/3/2017.
//

#include "ArithmeticJni.h"
#include "Arithmetic.h"
#include <ArithmeticInLib.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint JNICALL Java_com_sangwook_simplejni_ArithmeticJni_add(JNIEnv *env, jobject obj, jint lhs, jint rhs)
{
    jint result = add(lhs, rhs);
    return result;
}

JNIEXPORT jint JNICALL Java_com_sangwook_simplejni_ArithmeticJni_sub(JNIEnv *env, jobject obj, jint lhs, jint rhs)
{
    jint result = sub(lhs, rhs);
    return result;
}

JNIEXPORT jint JNICALL Java_com_sangwook_simplejni_ArithmeticJni_add_1in_1lib(JNIEnv *env, jobject obj, jint lhs, jint rhs)
{
    jint result = add_in_lib(lhs, rhs);
    return result;
}

JNIEXPORT jint JNICALL Java_com_sangwook_simplejni_ArithmeticJni_sub_1in_1lib(JNIEnv *env, jobject obj, jint lhs, jint rhs)
{
    jint result = sub_in_lib(lhs, rhs);
    return result;
}

#ifdef __cplusplus
}
#endif
