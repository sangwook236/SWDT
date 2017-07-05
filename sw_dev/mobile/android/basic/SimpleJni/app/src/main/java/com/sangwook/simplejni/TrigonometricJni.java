package com.sangwook.simplejni;

// REF [doc] >> JNI_in_Android.txt

public class TrigonometricJni
{
	public native double sin(double val);
	public native double cos(double val);

    static {
        System.loadLibrary("external_lib");
    }
}
