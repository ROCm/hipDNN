# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)
find_package(hip REQUIRED)

add_executable(public_hipdnn_backend_tests 
    test_backend_public_functions.cpp
    test_variant_pack_api.cpp
    test_handle_api.cpp
)

target_compile_options(public_hipdnn_backend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(public_hipdnn_backend_tests
    gtest 
    gtest_main 
    pthread 
    hipdnn_sdk
    hipdnn_backend
    hip::device
)

add_dependencies(check public_hipdnn_backend_tests)
clang_tidy_check(public_hipdnn_backend_tests)
gtest_discover_tests(
    public_hipdnn_backend_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_MODE PRE_TEST
)