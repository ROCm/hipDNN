# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

add_executable(public_hipdnn_backend_tests test_backend_public_functions.cpp)

target_link_libraries(public_hipdnn_backend_tests gtest gtest_main pthread hipdnn_backend)
target_include_directories(public_hipdnn_backend_tests PRIVATE 
    ${HIP_DNN_BACKEND_INCLUDE_DIR}
)

add_dependencies(check public_hipdnn_backend_tests)
gtest_discover_tests(
    public_hipdnn_backend_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_MODE PRE_TEST
)