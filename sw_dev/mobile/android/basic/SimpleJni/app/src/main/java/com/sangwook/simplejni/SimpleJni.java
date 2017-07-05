package com.sangwook.simplejni;

// REF [doc] >> JNI_in_Android.txt

public class SimpleJni {
    public native String getStringFromNative();

    static {
        System.loadLibrary("simple_native_lib");
    }
}
