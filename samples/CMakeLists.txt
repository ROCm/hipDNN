# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
project(hipdnn_samples VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(hip REQUIRED)
find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hipdnn_backend CONFIG REQUIRED)
find_package(hipdnn_sdk CONFIG REQUIRED)

add_executable(bn_inference
    batchnorm/bn_inference.cpp
)
target_link_libraries(bn_inference PRIVATE 
    hip::host
    hip::device
    hipdnn_frontend
    hipdnn_backend
    hipdnn_sdk
)

add_executable(bn_training
    batchnorm/bn_training.cpp
)
target_link_libraries(bn_training PRIVATE 
    hip::host
    hip::device
    hipdnn_frontend
    hipdnn_backend
    hipdnn_sdk
)

add_executable(bn_backward
    batchnorm/bn_backward.cpp
)
target_link_libraries(bn_backward PRIVATE 
    hip::host
    hip::device
    hipdnn_frontend
    hipdnn_backend
    hipdnn_sdk
)