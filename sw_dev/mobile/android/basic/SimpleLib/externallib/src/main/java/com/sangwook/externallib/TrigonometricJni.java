package com.sangwook.externallib;

// REF [doc] >> simplejni_usage_guide.txt
// REF [doc] >> android_usage_guide.txt

public class TrigonometricJni
{
	public native double sin(double val);
	public native double cos(double val);

    static {
        System.loadLibrary("external_lib");
    }
}
