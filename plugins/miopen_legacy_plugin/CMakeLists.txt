# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

add_compile_definitions(__HIP_PLATFORM_AMD__)

set(MIOPEN_LEGACY_PLUGIN "0.0.1")

if(NOT WIN32)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")
endif()

project(
    hipDNN
    VERSION ${MIOPEN_LEGACY_PLUGIN}
    LANGUAGES CXX)
    
set(CMAKE_CXX_STANDARD 20)

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

find_package(hip REQUIRED)
find_package(hipdnn_sdk CONFIG REQUIRED)
find_package(miopen CONFIG REQUIRED)


add_library(miopen_legacy_plugin SHARED
    miopen_legacy_plugin.cpp
)

target_link_libraries(miopen_legacy_plugin PUBLIC MIOpen hipdnn_sdk)
