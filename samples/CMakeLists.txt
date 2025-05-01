# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
project(hipdnn_samples VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(hip REQUIRED)

find_package(hipdnn_frontend CONFIG REQUIRED)

add_executable(hipdnn_samples 
    main.cpp
)
target_link_libraries(hipdnn_samples PUBLIC hipdnn_frontend) 
