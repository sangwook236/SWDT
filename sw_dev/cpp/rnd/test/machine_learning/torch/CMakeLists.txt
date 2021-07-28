# Usage:
#    mkdir build
#    cd build
#    cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
#    cmake --build . --config Release

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(my_torch_project)

#find_package(Threads REQUIRED)
#find_package(Boost 1.62 REQUIRED thread timer)
#find_package(Eigen 3.2 REQUIRED)
#find_package(Log4cxx REQUIRED)
#find_package(Qt5 COMPONENTS Core Gui Widgets Quick Qml Concurrent REQUIRED)  # target_link_libraries(myexe Qt5::Qt5Core Qt5::Qt5Widgets)
#find_package(GSL REQUIRED)
#find_package(OpenCV2 4.2.0 REQUIRED)
#find_package(CUDA)
find_package(Torch REQUIRED)

#include_directories(${Boost_INCLUDE_DIRS})
#include_directories(${EIGEN_INCLUDE_DIRS})
#include_directories(${LOG4CXX_INCLUDE_DIR})
#include_directories(${QT_INCLUDE_DIR})
#include_directories(${GSL_INCLUDE_DIRS})
#include_directories(${OpenCV2_INCLUDE_DIRS})
#include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

#link_directories(${Boost_LIBRARY_DIRS})

#add_definitions(-D_DEBUG)
#add_definitions(${QT_DEFINITIONS})

set(TARGET torch_test)

set(SRCS
    torch_test.cpp
)
set(LIBS
    #${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
)

add_executable(${TARGET} ${SRCS})
#cuda_add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 14)
