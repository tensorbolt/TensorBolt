cmake_minimum_required(VERSION 2.6)
set(CMAKE_BUILD_TYPE Debug)
project(tensorbolt)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/dependencies")

add_subdirectory(ndarray)
add_subdirectory(tb_graph)
add_subdirectory(unittest)

#set_property(tensorbolt cgraph PROPERTY POSITION_INDEPENDENT_CODE ON)
