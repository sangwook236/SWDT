# Usage:
#   Configure and generate:
#       mkdir build
#       cd build
#       cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
#   Build:
#       cd ..
#       cmake --build ./build --config Release --target all -j 14 --
#   Run:
#       ./build/torch_test

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(my_torch_project VERSION 0.1.0)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
#set(CMAKE_PREFIX_PATH "/path/to/libtorch")

#set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

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

add_definitions(-DUNICODE -D_UNICODE)
#add_definitions(-D_DEBUG -DUNICODE -D_UNICODE)
#add_definitions(${QT_DEFINITIONS})

#if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
#	add_compile_options(-fPIC)
#endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

set(TARGET torch_test)

set(SRCS
    torch_test.cpp
)
set(LIBS
    #${LOG4CXX_LIBRARIES}
    #Qt5::Core Qt5::Widgets
    #${OpenCV_LIBRARIES}
    #${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
)

add_executable(${TARGET} ${SRCS})
#cuda_add_executable(${TARGET} ${SRCS})
#add_library(${TARGET} SHARED ${SRCS})
#add_library(${TARGET} STATIC ${SRCS})

target_link_libraries(${TARGET} ${LIBS})
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 17)
