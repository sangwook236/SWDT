# Usage:
#	Configure and generate:
#		mkdir build
#		cd build
#		cmake -DCMAKE_PREFIX_PATH=/path/to/lib ..
#	Build:
#		cd ..
#		cmake --build ./build --config Release --target all -j 14 --
#	Run:
#		./build/my_cpp_test

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(my_cpp_test VERSION 0.1.0)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
#set(CMAKE_PREFIX_PATH "/path/to/lib")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 17)
endif()
#set(CMAKE_DEBUG_POSTFIX d)

if(APPLE)
elseif(CYGWIN)
elseif(MSYS OR MINGW)
elseif(UNIX AND NOT APPLE AND NOT CYGWIN)
	set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
#	set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
elseif(WIN32 AND NOT CYGWIN)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
elseif(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#	set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -Wno-deprecated-declarations")
	set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
#	set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
#	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

	add_definitions(-DUNICODE -D_UNICODE)

#	add_compile_options(-fPIC)
#	add_compile_options(-fpermissive)
#	add_compile_options(-Wno-strict-aliasing)
#	add_compile_options(-Wno-unused-function)
#	add_compile_options(-Wno-unused-variable)
#	add_compile_options(-Wno-unused-parameter)
elseif(CMAKE_CXX_COMPILER_ID MATCHES Intel)
elseif(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
	set(CMAKE_CXX_FLAGS "/Wall")
#	set(CMAKE_CXX_FLAGS "/std:c++17 /Wall")
	set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Ob0 /Od /RTC1 /D_DEBUG")
	set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Ob2 /D NDEBUG")
#	set(CMAKE_CXX_FLAGS_MINSIZEREL "/MD /O1 /Ob1 /D NDEBUG")
#	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /Zi /O2 /Ob1 /D NDEBUG")

	add_definitions(/DUNICODE /D_UNICODE)
endif()

#--------------------
find_package(Threads REQUIRED)
#find_package(OpenMP REQUIRED)
#find_package(Boost 1.62 REQUIRED thread timer)
#find_package(Log4cxx REQUIRED)
#find_package(glog REQUIRED)
#find_package(gflags REQUIRED)
#find_package(ZLIB REQUIRED)
#find_package(CUDA REQUIRED)
#find_package(GSL REQUIRED)
#find_package(Qt5 COMPONENTS Core Gui Widgets Quick Qml Concurrent REQUIRED)
#find_package(OpenCV 4.2.0 REQUIRED)
#find_package(PCL REQUIRED)
#find_package(Open3D REQUIRED)
find_package(Ceres REQUIRED)

#find_library(MY_LIBRARIES NAMES libmy.a)
#find_library(MY_LIBRARIES NAMES libmy.a PATHS /path/to/lib)
#message(STATUS "MY_LIBRARIES found: ${MY_LIBRARIES}")

#include_directories(${Boost_INCLUDE_DIRS})
#include_directories(${LOG4CXX_INCLUDE_DIR})
#include_directories(${ZLIB_INCLUDE_DIRS})
#include_directories(${CUDA_INCLUDE_DIRS})
#include_directories(/usr/include/eigen3)
#include_directories(${GSL_INCLUDE_DIRS})
#include_directories(${QT_INCLUDE_DIR})
#include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${PCL_INCLUDE_DIRS})
#include_directories(${Open3D_INCLUDE_DIRS})
include_directories(/usr/local/include)
#message(STATUS "Include directories: ${INCLUDE_DIRECTORIES}")

#link_directories(${Boost_LIBRARY_DIRS})
#link_directories(BEFORE /usr/local/lib)
#message(STATUS "Link directories: ${LINK_DIRECTORIES}")

#add_definitions(${OpenMP_CXX_FLAGS})
#add_definitions(${QT_DEFINITIONS})
#message(STATUS "Compile definitions: ${COMPILE_DEFINITIONS}")

if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
elseif(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	if(OpenMP_CXX_FOUND)
		add_compile_options(-fopenmp)
	endif(OpenMP_CXX_FOUND)
elseif(CMAKE_CXX_COMPILER_ID MATCHES Intel)
elseif(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
	if(OpenMP_CXX_FOUND)
		add_compile_options(/fopenmp)
	endif(OpenMP_CXX_FOUND)
endif()

#--------------------
#set(TARGET my_cpp_test)
set(TARGET ${CMAKE_PROJECT_NAME})  # The name of the top level project.
#set(TARGET ${PROJECT_NAME})  # The name of the project given to the project command.

set(SRCS
	ceres_solver_main.cpp
	ceres_solver_bundle_adjustment_example.cpp
	ceres_solver_curve_fitting_example.cpp
	bal_problem.cpp
)
set(LIBS
#	${OpenMP_CXX_FLAGS}
#	OpenMP::OpenMP_CXX
#	${LOG4CXX_LIBRARIES}
#	${GLOG_LIBRARIES}
#	${GFLAGS_LIBRARIES}
#	${ZLIB_LIBRARIES}
#	${CUDA_LIBRARIES}
#	Qt5::Core Qt5::Widgets
#	${OpenCV_LIBRARIES}
#	${PCL_LIBRARIES}
#	${Open3D_LIBRARIES}
	${CERES_LIBRARIES}
)

add_executable(${TARGET} ${SRCS})
#add_library(${TARGET} SHARED ${SRCS})
#add_library(${TARGET} STATIC ${SRCS})
#cuda_add_executable(${TARGET} ${SRCS})
#cuda_add_library(${TARGET} SHARED ${SRCS})
#cuda_add_library(${TARGET} STATIC ${SRCS})

target_link_libraries(${TARGET} ${LIBS})
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)
