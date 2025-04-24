# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
project(hipdnn_samples VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(hip REQUIRED)

# NOTE: Workaround until llvm & hip cmake modules fixes symlink logic in their config files; remove when fixed.
#       Added to prevent issues with not finding hip cmake.
list(
    APPEND
    CMAKE_PREFIX_PATH
    ${ROCM_PATH}/llvm
    ${ROCM_PATH}
    ${ROCM_PATH}/hip
    /opt/rocm/llvm
    /opt/rocm
    /opt/rocm/hip 
)

add_executable(hipdnn_samples 
    main.cpp
)

target_include_directories(hipdnn_samples SYSTEM PRIVATE ${HIP_INCLUDE_DIRS})
