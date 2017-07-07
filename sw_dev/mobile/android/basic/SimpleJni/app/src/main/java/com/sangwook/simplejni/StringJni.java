package com.sangwook.simplejni;

// REF [doc] >> simplejni_usage_guide.txt
// REF [doc] >> android_usage_guide.txt

public class StringJni {
    public native String getStringFromNative();

    static {
        System.loadLibrary("string_lib");
    }
}
