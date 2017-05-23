package javacpp;

import com.googlecode.javacpp.*;
import com.googlecode.javacpp.annotation.*;

@Platform(
	// For linking static libraries.
	include = {
	    "LegacyLibrary.h"
	},
	includepath={
		//"../LegacyLibrary_static",
		"D:/work_center/sw_dev/java/ext/test/native_interface/LegacyLibrary_static",
		"D:/work_center/sw_dev/cpp/ext/inc",
		"D:/work_center/sw_dev/cpp/rnd/inc",
		"C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0A/Include"
	},
	link={
		"LegacyLibrary"
	},
	linkpath = {
		//"../LegacyLibrary_static/x64/Release",
		"D:/work_center/sw_dev/java/ext/test/LegacyLibrary_static/x64/Release",
		"D:/work_center/sw_dev/java/ext/test/native_interface/LegacyLibrary_static/x64/Release",
		"D:/work_center/sw_dev/cpp/ext/lib",
		"D:/work_center/sw_dev/cpp/rnd/lib"
	}

/*
	// For linking shared libraries.
	// Shared library files exist in the same directory as a java source file
	include = {
	    "LegacyLibrary.h"
	},
	includepath={
		"../../LegacyLibrary_shared",
		"D:/work_center/sw_dev/cpp/ext/inc",
		"D:/work_center/sw_dev/cpp/rnd/inc",
		"C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0A/Include"
	},
	link={
		"LegacyLibrary"
	},
	linkpath = {
		"../../LegacyLibrary_shared/x64/Release",
		"D:/work_center/sw_dev/cpp/ext/lib",
		"D:/work_center/sw_dev/cpp/rnd/lib"
	}
*/
)

@Namespace("LegacyLibrary")
public class LegacyLibrary {
    public static class LegacyClass extends Pointer {
        static { Loader.load(); }
        public LegacyClass() { allocate(); }
        private native void allocate();

        // to call the getter and setter functions 
        public native @StdString String get_property();
        public native void set_property(String property);

        // to access the member variable directly
        public native @StdString String property();
        public native void property(String property);
    }

    public static void run(String[] args)
    {
        try
        {
        	java.io.File dir1 = new java.io.File(".");
        	java.io.File dir2 = new java.io.File("..");
        	
            System.out.println("Current dir : " + dir1.getCanonicalPath());
            System.out.println("Parent  dir : " + dir2.getCanonicalPath());
            
            // Pointer objects allocated in Java get deallocated once they become unreachable,
            // but C++ destructors can still be called in a timely fashion with Pointer.deallocate().
            LegacyClass l = new LegacyClass();
            l.set_property("Hello World!");
            System.out.println(l.property());
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }

    public static void main(String[] args)
    {
    	run(args);
    }
}