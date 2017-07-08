package com.sangwook.simpleapp;

// REF [doc] >> simplejni_usage_guide.txt
// REF [doc] >> android_usage_guide.txt

public class ArithmeticJni
{
	public native int add(int lhs, int rhs);
	public native int sub(int lhs, int rhs);

    static {
        System.loadLibrary("arithmetic_lib");
    }
}
