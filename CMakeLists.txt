cmake_minimum_required (VERSION 2.8) 
project (FastCamTrack) 

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unknown-pragmas -msse4  -mavx  -std=c++11") # -Wall

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wno-unknown-pragmas -Wno-deprecated-declarations -Wall -msse4  -mavx  -std=c++11") # -Wall

#cmake ../ -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3

FIND_PACKAGE(OpenCV REQUIRED)
#FIND_PACKAGE(Eigen3 REQUIRED) 
include_directories(${EIGEN3_INCLUDE_DIR})

set(CODEFILES run_io_reprojection_test.cpp utilities.cpp camera.cpp pose.cpp odometer.cpp)

# GrayScale, Optical Flow
add_executable (run_io_reprojection_test ${CODEFILES})
TARGET_LINK_LIBRARIES(run_io_reprojection_test ${OpenCV_LIBS})

