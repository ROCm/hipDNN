# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(HIP_DNN_BACKEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "Path to the HIP DNN backend include directory")
set(HIP_DNN_BACKEND_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR "${PROJECT_BINARY_DIR}/backend/include/")

add_subdirectory(src)
add_subdirectory(tests)