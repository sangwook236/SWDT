package com.sangwook.simplejni;

// REF [doc] >> JNI_in_Android.txt

public class ArithmeticJni
{
	public native int add(int lhs, int rhs);
	public native int sub(int lhs, int rhs);

    static {
        System.loadLibrary("arithmetic_lib");
    }
}
