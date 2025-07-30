# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Frontend Unit tests")
    return()
endif()

find_package(hip REQUIRED)

add_executable(hipdnn_frontend_tests 
    main.cpp
    fake_backend/fake_hipdnn_backend.cpp
    test_backend_logging_helpers.cpp
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
    test_scoped_hipdnn_backend_descriptor.cpp
    test_tensor_attributes.cpp
    test_tensor_value_attributes.cpp
    test_types.cpp
    test_utilities.cpp
)

target_compile_options(hipdnn_frontend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_compile_definitions(hipdnn_frontend_tests PRIVATE
    COMPONENT_NAME="hipdnn_frontend_tests"
)

target_link_libraries(hipdnn_frontend_tests 
    GTest::gtest 
    GTest::gmock
    hip::host
    pthread 
    hipdnn_sdk 
    hipdnn_frontend
)
clang_tidy_check(hipdnn_frontend_tests)

add_dependencies(hipdnn_frontend_tests generate_hipdnn_sdk_headers)

add_target_to_check_targets(hipdnn_frontend_tests ${CMAKE_CURRENT_BINARY_DIR})
