package com.sangwook.samplejni;

// REF [doc] >> JNI_in_Android.txt

public class SampleJni {
    public native String getStringFromNative();

    static {
        System.loadLibrary("sample_native_lib");
    }
}
