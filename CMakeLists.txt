cmake_minimum_required(VERSION 2.8)
if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" VERSION_GREATER  3.1)
cmake_policy(SET CMP0054 OLD)
endif("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" VERSION_GREATER  3.1)

include("cmake/yong_injector/injector.cmake")
include("cmake/yong_hsprj.cmake")
include("cmake/yong_hslib.cmake")

hs_project(hs_feature2d)

hs_parse_dependencies("dependencies.txt")

add_subdirectory(include)
add_subdirectory("src")
add_subdirectory("unit_test")

hs_project_end(hs_feature2d)
