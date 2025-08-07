# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)
find_package(hip REQUIRED)

add_executable(public_hipdnn_frontend_tests 
    main.cpp
    frontend_integration_test.cpp
)

target_compile_options(public_hipdnn_frontend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(public_hipdnn_frontend_tests
    GTest::gtest 
    GTest::gtest_main 
    pthread 
    hipdnn_sdk
    hipdnn_backend
    hipdnn_frontend
    hip::host
    hip::device
)

target_include_directories(public_hipdnn_frontend_tests
    PRIVATE
    ${HIPDNN_TEST_INCLUDE_DIRS}
)

# Define test plugin directory
target_compile_definitions(public_hipdnn_frontend_tests
    PRIVATE
    TEST_PLUGIN_DIR="${CMAKE_BINARY_DIR}/tests/test_plugins"
)

# Ensure test_good_plugin is built before frontend tests
add_dependencies(public_hipdnn_frontend_tests test_good_plugin test_execute_fails_plugin test_no_applicable_engines_plugin)

clang_tidy_check(public_hipdnn_frontend_tests)
add_target_to_check_targets(public_hipdnn_frontend_tests ${CMAKE_CURRENT_BINARY_DIR})
