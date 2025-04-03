
################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

cmake_minimum_required(VERSION 3.21...3.22)


set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/cmake/ClangToolChain.cmake" CACHE STRING "Toolchain file")
set(HIPDNN_VERSION "0.0.1")
project(
    hipDNN
    VERSION ${HIPDNN_VERSION}
    LANGUAGES CXX)
    
set(CMAKE_CXX_STANDARD 17)

list(
    APPEND
    CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${ROCM_PATH}/lib/cmake/hip
    /opt/rocm/lib/cmake/hip
    ${HIP_DIR}/cmake
)


include(cmake/Dependencies.cmake)
hipdnn_add_dependency(
    ROCmCMakeBuildTools
    VERSION 0.11.0
    FIND_PACKAGE_ARGS NO_DEFAULT_PATH # PATHS "${ROCM_PATH}"
)

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMClients)

option(HIP_DNN_BUILD_BACKEND "Build the backend code" ON)
option(HIP_DNN_BUILD_FRONTEND "Build the frontend code" ON)

# Add global compile options
# fPIC: generates position-independent code,
add_compile_options(-fPIC)
if(HIP_DNN_BUILD_BACKEND)
    add_subdirectory(backend)
endif()

if(HIP_DNN_BUILD_FRONTEND)
    add_subdirectory(frontend)
endif()

add_subdirectory(tests)