# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
project(hipdnn_samples VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(hip REQUIRED)
find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hipdnn_backend CONFIG REQUIRED)
find_package(hipdnn_sdk CONFIG REQUIRED)

add_executable(hipdnn_samples 
    main.cpp
)

add_executable(bn_inference
    bn_inference.cpp
)
target_link_libraries(bn_inference PRIVATE 
    hip::host 
    hipdnn_frontend
    hipdnn_backend
)

add_executable(bn_training
    bn_training.cpp
)
target_link_libraries(bn_training PRIVATE 
    hip::host 
    hipdnn_frontend
    hipdnn_backend
)

add_executable(bn_backward
    bn_backward.cpp
)
target_link_libraries(bn_backward PRIVATE 
    hip::host 
    hipdnn_frontend
    hipdnn_backend
)

target_compile_definitions(hipdnn_samples PUBLIC
    __HIP_PLATFORM_AMD__
)

target_link_libraries(hipdnn_samples PUBLIC hip::host)
target_link_libraries(hipdnn_samples PUBLIC hipdnn_frontend) 
target_link_libraries(hipdnn_samples PUBLIC hipdnn_backend)
