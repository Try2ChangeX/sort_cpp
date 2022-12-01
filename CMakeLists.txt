cmake_minimum_required(VERSION 3.23)
project(sort_cpp)

set(CMAKE_CXX_STANDARD 14)
#set(OpenCV_DIR "/usr/local/opencv")
find_package(OpenCV REQUIRED)

include_directories(.)

add_executable(sort_cpp
        Hungarian.cpp
        Hungarian.h
        KalmanTracker.cpp
        KalmanTracker.h
        main.cpp
        Sort.cpp
        Sort.h)

target_link_libraries(sort_cpp ${OpenCV_LIBS})
