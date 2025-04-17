# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Frontend Unit tests")
    return()
endif()

include(GoogleTest)

add_executable(hipdnn_frontend_tests 
    test_pointwise_node.cpp
    test_pointwise_attributes.cpp
    test_batchnorm_inference_node.cpp
    test_batchnorm_backward_node.cpp
    test_batchnorm_inference_attributes.cpp
    test_batchnorm_backward_attributes.cpp
    test_graph.cpp
    test_utilities.cpp
    test_graph_generated.cpp
)

target_link_libraries(hipdnn_frontend_tests gtest gtest_main pthread flatbuffers)
target_include_directories(hipdnn_frontend_tests PRIVATE 
    ${HIP_DNN_FRONTEND_INCLUDE_DIR}
    ${HIP_DNN_SDK_INCLUDE_DIR}
)
clang_tidy_check(hipdnn_frontend_tests)

add_dependencies(hipdnn_frontend_tests generate_hipdnn_sdk_headers)
add_dependencies(check hipdnn_frontend_tests)
gtest_discover_tests(
    hipdnn_frontend_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_MODE PRE_TEST
)
