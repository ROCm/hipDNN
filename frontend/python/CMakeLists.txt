# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(hipdnn
    MODULE hipdnn.cpp
           attributes.cpp
           types.cpp
           pygraph/pygraph.cpp
           pygraph/norm.cpp
           pygraph/pointwise.cpp
    WITH_SOABI)
target_link_libraries(hipdnn PRIVATE pybind11::headers)
target_link_libraries(hipdnn INTERFACE hipdnn_frontend hipdnn_sdk)

target_include_directories(hipdnn PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../frontend/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../sdk/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../build/_deps/flatbuffers-src/include)

target_compile_definitions(hipdnn PRIVATE)

add_subdirectory(tests)