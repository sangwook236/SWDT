# Usage:
#	Configure and generate:
#		mkdir build
#		cd build
#		cmake -DCMAKE_PREFIX_PATH=/path/to/lib ..
#	Build:
#		cd ..
#		cmake --build ./build --config Release --target all -j 14 --
#	Run:
#		./build/test

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(my_cpp_test VERSION 0.1.0)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
#set(CMAKE_PREFIX_PATH "/path/to/lib")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
#set(CMAKE_DEBUG_POSTFIX d)

if(APPLE)
elseif(CYGWIN)
elseif(MSYS OR MINGW)
elseif(UNIX AND NOT APPLE AND NOT CYGWIN)
	set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
	set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
elseif(WIN32 AND NOT CYGWIN)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
elseif(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
	set(CMAKE_CXX_FLAGS_DEBUG "-g")
	set(CMAKE_CXX_FLAGS_RELEASE "-O2")
elseif(CMAKE_CXX_COMPILER_ID MATCHES Intel)
elseif(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
	set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Ob0 /Od /RTC1 /D_DEBUG")
#	set(CMAKE_CXX_FLAGS_MINSIZEREL "/MD /O1 /Ob1 /D NDEBUG")
	set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Ob2 /D NDEBUG")
#	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /Zi /O2 /Ob1 /D NDEBUG")
endif()

#--------------------
find_package(Threads REQUIRED)
#find_package(OpenMP REQUIRED)
#find_package(Boost 1.62 REQUIRED thread timer)
#find_package(Log4cxx REQUIRED)
#find_package(ZLIB REQUIRED)
#find_package(GSL REQUIRED)
#find_package(Qt5 COMPONENTS Core Gui Widgets Quick Qml Concurrent REQUIRED)
#find_package(OpenCV 4.2.0 REQUIRED)
#find_package(CUDA REQUIRED)

#include_directories(${Boost_INCLUDE_DIRS})
#include_directories(${LOG4CXX_INCLUDE_DIR})
#include_directories(${ZLIB_INCLUDE_DIRS})
#include_directories(/usr/include/eigen3)
#include_directories(${GSL_INCLUDE_DIRS})
#include_directories(${QT_INCLUDE_DIR})
#include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${CUDA_INCLUDE_DIRS})

#link_directories(${Boost_LIBRARY_DIRS})

add_definitions(-DUNICODE -D_UNICODE)
#add_definitions(${QT_DEFINITIONS})

#if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
#	add_compile_options(-fPIC)
#endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)
if(OpenMP_CXX_FOUND)
	add_compile_options(-fopenmp)
endif(OpenMP_CXX_FOUND)

#--------------------
set(TARGET test)

set(SRCS
	main.cpp
)
set(LIBS
#	${OpenMP_CXX_FLAGS}
#	${LOG4CXX_LIBRARIES}
#	${ZLIB_LIBRARIES}
#	Qt5::Core Qt5::Widgets
#	${OpenCV_LIBRARIES}
#	${CUDA_LIBRARIES}
)

add_executable(${TARGET} ${SRCS})
#add_library(${TARGET} SHARED ${SRCS})
#add_library(${TARGET} STATIC ${SRCS})
#cuda_add_executable(${TARGET} ${SRCS})
#cuda_add_library(${TARGET} SHARED ${SRCS})
#cuda_add_library(${TARGET} STATIC ${SRCS})

target_link_libraries(${TARGET} ${LIBS})
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)
