package com.sangwook.simplejni;

// REF [doc] >> JNI_in_Android.txt

public class StringJni {
    public native String getStringFromNative();

    static {
        System.loadLibrary("string_lib");
    }
}
