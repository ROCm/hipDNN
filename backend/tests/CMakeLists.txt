# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# TODO Do not enable HIP language globally, use HIP RTC instead.
# The library does not contain any HIP code (or any other device code).
# It is required only for hipdnn_test_engine_plugin1.
enable_language(HIP)

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
    test_handle.cpp
    test_helpers.cpp
    test_plugin.cpp
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

# libhipdnn_test_plugin1.so or hipdnn_test_plugin1.dll (used by test_plugin.cpp)
add_library(hipdnn_test_plugin1 SHARED plugins/plugin1.cpp)
target_link_libraries(hipdnn_test_plugin1 PRIVATE hipdnn_sdk)
set_target_properties(hipdnn_test_plugin1 PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_options(hipdnn_test_plugin1 PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
clang_tidy_check(hipdnn_test_plugin1)
add_dependencies(hipdnn_backend_tests hipdnn_test_plugin1)

# libhipdnn_test_plugin2.so or hipdnn_test_plugin2.dll (used by test_plugin.cpp)
add_library(hipdnn_test_plugin2 SHARED plugins/plugin2.cpp)
target_link_libraries(hipdnn_test_plugin2 PRIVATE hipdnn_sdk)
set_target_properties(hipdnn_test_plugin2 PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_options(hipdnn_test_plugin2 PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
clang_tidy_check(hipdnn_test_plugin2)
add_dependencies(hipdnn_backend_tests hipdnn_test_plugin2)

# libhipdnn_test_engine_plugin1.so or hipdnn_test_engine_plugin1.dll (used by test_engine_plugin.cpp)
add_library(hipdnn_test_engine_plugin1 SHARED plugins/engine_plugin1.cpp)
set_source_files_properties(plugins/engine_plugin1.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(hipdnn_test_engine_plugin1 PRIVATE hipdnn_sdk)
set_target_properties(hipdnn_test_engine_plugin1 PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_options(hipdnn_test_engine_plugin1 PRIVATE --offload-arch=gfx90a --offload-arch=gfx942 --offload-arch=gfx1100 --offload-arch=gfx1151 --offload-arch=gfx1201)
target_compile_options(hipdnn_test_engine_plugin1 PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
clang_tidy_check(hipdnn_test_engine_plugin1)
add_dependencies(hipdnn_backend_tests hipdnn_test_engine_plugin1)

add_dependencies(check hipdnn_backend_tests)
gtest_discover_tests(
    hipdnn_backend_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_MODE PRE_TEST
)
