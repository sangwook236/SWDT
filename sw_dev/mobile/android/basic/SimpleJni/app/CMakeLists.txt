# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
             string_lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/StringJni.cpp )

add_library( # Sets the name of the library.
             arithmetic_lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/Arithmetic.cpp
             src/main/cpp/ArithmeticJni.cpp )

#add_library(external_arithmetic_lib
#            SHARED
#            IMPORTED)
#set_target_properties(external_arithmetic_lib
#                      PROPERTIES IMPORTED_LOCATION libexternal_lib.so )
#                      #PROPERTIES IMPORTED_LOCATION jnilibs/bin/${CMAKE_BUILD_TYPE}/obj/${ANDROID_ABI}/libexternal_lib.so )
#                      #PROPERTIES IMPORTED_LOCATION D:/work/SWDT_github/sw_dev/mobile/android/basic/SimpleLib/externallib/build/intermediates/cmake/${CMAKE_BUILD_TYPE}/obj/${ANDROID_ABI}/libexternal_lib.so )
#                      #PROPERTIES IMPORTED_LOCATION D:/work/SWDT_github/sw_dev/mobile/android/basic/SimpleJni/app/jnilibs/bin/${CMAKE_BUILD_TYPE}/obj/${ANDROID_ABI}/libexternal_lib.so )
#include_directories(jnilibs/include)
#link_directories(jnilibs/bin)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       string_lib

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )

target_link_libraries( # Specifies the target library.
                       arithmetic_lib

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )
