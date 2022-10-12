# Usage:
#	Configure and generate:
#		mkdir build
#		cd build
#		cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
#	Build:
#		cd ..
#		cmake --build ./build --config Release --target all -j 14 --
#	Run:
#		./build/torch_runner

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch_runner VERSION 0.1.0)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
#set(CMAKE_PREFIX_PATH "/path/to/libtorch")

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
	set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
#	set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
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
#find_package(Threads REQUIRED)
#find_package(Boost 1.62 REQUIRED thread timer)
#find_package(Log4cxx REQUIRED)
#find_package(Eigen 3.2 REQUIRED)
#find_package(GSL REQUIRED)
#find_package(Qt5 COMPONENTS Core Gui Widgets Quick Qml Concurrent REQUIRED)
#find_package(OpenCV 4.2.0 REQUIRED)
#find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

#include_directories(${Boost_INCLUDE_DIRS})
#include_directories(${LOG4CXX_INCLUDE_DIR})
#include_directories(${EIGEN_INCLUDE_DIRS})
#include_directories(${GSL_INCLUDE_DIRS})
#include_directories(${QT_INCLUDE_DIR})
#include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

#link_directories(${Boost_LIBRARY_DIRS})

#add_definitions(${OpenMP_CXX_FLAGS})
#add_definitions(${QT_DEFINITIONS})

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
#set(TARGET torch_runner)
set(TARGET ${CMAKE_PROJECT_NAME})  # The name of the top level project.
#set(TARGET ${PROJECT_NAME})  # The name of the project given to the project command.

set(SRCS
	torch_runner.cpp
	torch_main.cpp
	torch_basic.cpp
	torch_torch_script.cpp
	torch_training.cpp
)
set(LIBS
#	${OpenMP_CXX_FLAGS}
#	OpenMP::OpenMP_CXX
#	${LOG4CXX_LIBRARIES}
#	${ZLIB_LIBRARIES}
#	Qt5::Core Qt5::Widgets
#	${OpenCV_LIBRARIES}
#	${CUDA_LIBRARIES}
	${TORCH_LIBRARIES}
)

add_executable(${TARGET} ${SRCS})
#add_library(${TARGET} SHARED ${SRCS})
#add_library(${TARGET} STATIC ${SRCS})
#cuda_add_executable(${TARGET} ${SRCS})
#cuda_add_library(${TARGET} SHARED ${SRCS})
#cuda_add_library(${TARGET} STATIC ${SRCS})

target_link_libraries(${TARGET} ${LIBS})
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)
