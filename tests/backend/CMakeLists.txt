# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

add_executable(public_hipdnn_backend_tests 
    test_backend_public_functions.cpp
)

target_compile_options(public_hipdnn_backend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(public_hipdnn_backend_tests
    gtest 
    gtest_main 
    pthread 
    hipdnn_sdk
    hipdnn_backend
)

add_dependencies(check public_hipdnn_backend_tests)
clang_tidy_check(public_hipdnn_backend_tests)
gtest_discover_tests(
    public_hipdnn_backend_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_MODE PRE_TEST
)