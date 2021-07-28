# Usage:
#    mkdir build
#    cd build
#    cmake -DCMAKE_PREFIX_PATH=/path/to/lib ..
#    cmake --build . --config Release

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(my_cpp_test)

find_package(Threads REQUIRED)
#find_package(Boost 1.62 REQUIRED thread timer)
#find_package(Log4cxx REQUIRED)
#find_package(Eigen 3.2 REQUIRED)
#find_package(GSL REQUIRED)
#find_package(Qt5 COMPONENTS Core Gui Widgets Quick Qml Concurrent REQUIRED)
#find_package(OpenCV2 4.2.0 REQUIRED)
#find_package(CUDA)

#include_directories(${Boost_INCLUDE_DIRS})
#include_directories(${LOG4CXX_INCLUDE_DIR})
#include_directories(${EIGEN_INCLUDE_DIRS})
#include_directories(${GSL_INCLUDE_DIRS})
#include_directories(${QT_INCLUDE_DIR})
#include_directories(${OpenCV2_INCLUDE_DIRS})
#include_directories(${CUDA_INCLUDE_DIRS})

#link_directories(${Boost_LIBRARY_DIRS})

#add_definitions(-D_DEBUG)
#add_definitions(${QT_DEFINITIONS})

#if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
#	add_compile_options(-fPIC)
#endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

set(TARGET test)

set(SRCS
    main.cpp
)
set(LIBS
    #${LOG4CXX_LIBRARIES}
    #Qt5::Core Qt5::Widgets
    #${OpenCV2_LIBRARIES}
    #${CUDA_LIBRARIES}
)

add_executable(${TARGET} ${SRCS})
#cuda_add_executable(${TARGET} ${SRCS})
#add_library(${TARGET} SHARED ${SRCS})
#add_library(${TARGET} STATIC ${SRCS})

target_link_libraries(${TARGET} ${LIBS})
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 14)
