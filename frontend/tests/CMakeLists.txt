# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Frontend Unit tests")
    return()
endif()

add_executable(hipdnn_frontend_tests 
    test_batchnorm_attributes.cpp
    test_batchnorm_backward_attributes.cpp
    test_batchnorm_backward_node.cpp
    test_batchnorm_inference_attributes.cpp
    test_batchnorm_inference_node.cpp
    test_batchnorm_node.cpp
    test_error.cpp
    test_graph.cpp
    test_graph_generated.cpp
    test_pointwise_attributes.cpp
    test_pointwise_node.cpp
    test_tensor_attributes.cpp
    test_types.cpp
    test_utilities.cpp
)

target_compile_options(hipdnn_frontend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(hipdnn_frontend_tests 
    GTest::gtest 
    GTest::gtest_main 
    pthread 
    hipdnn_sdk 
    hipdnn_frontend
)
clang_tidy_check(hipdnn_frontend_tests)

add_dependencies(hipdnn_frontend_tests generate_hipdnn_sdk_headers)
add_dependencies(check hipdnn_frontend_tests)
gtest_discover_tests(
    hipdnn_frontend_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_MODE PRE_TEST
)
