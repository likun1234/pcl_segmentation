cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(main)

find_package(PCL 1.2 REQUIRED)
add_compile_options(-std=c++11)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (main main.cpp)
target_link_libraries (main ${PCL_LIBRARIES})
