# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Backend Unit tests")
    return()
endif()

include(GoogleTest)

add_executable(hipdnn_backend_tests 
    test_hello_world.cpp
)

target_link_libraries(hipdnn_backend_tests 
    gtest 
    gtest_main 
    pthread 
    hipdnn_backend_private)
    target_include_directories(hipdnn_backend_tests 
    PRIVATE 
    ${HIP_DNN_BACKEND_SRC_DIR}
    ${HIP_DNN_BACKEND_INCLUDE_DIR}
    )
    
clang_tidy_check(hipdnn_backend_tests)

add_dependencies(check hipdnn_backend_tests)
gtest_discover_tests(
    hipdnn_backend_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_MODE PRE_TEST
)