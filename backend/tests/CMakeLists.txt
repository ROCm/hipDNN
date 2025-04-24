# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Backend Unit tests")
    return()
endif()

include(GoogleTest)

add_executable(hipdnn_backend_tests 
    descriptors/test_graph_descriptor.cpp
    descriptors/test_descriptor_factory.cpp
    test_engine_plugin.cpp
    test_error.cpp
    test_flatbuffer_utilities.cpp
    test_helpers.cpp
)

target_link_libraries(hipdnn_backend_tests 
    gtest 
    gtest_main 
    pthread 
    hipdnn_backend_private
    hipdnn_sdk
)
target_compile_options(hipdnn_backend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
clang_tidy_check(hipdnn_backend_tests)

add_library(engineplugin1 SHARED plugins/engine_plugin1.cpp)
target_link_libraries(engineplugin1 PRIVATE hipdnn_sdk)
add_dependencies(hipdnn_backend_tests engineplugin1)

add_library(engineplugin2 SHARED plugins/engine_plugin2.cpp)
target_link_libraries(engineplugin2 PRIVATE hipdnn_sdk)
add_dependencies(hipdnn_backend_tests engineplugin2)

add_dependencies(check hipdnn_backend_tests)
gtest_discover_tests(
    hipdnn_backend_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_MODE PRE_TEST
)