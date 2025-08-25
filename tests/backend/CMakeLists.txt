# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

add_executable(public_hipdnn_backend_tests 
    test_plugin_loading.cpp
    test_backend_public_functions.cpp
    test_backend_execute_api.cpp
    test_engine_config_api.cpp
    test_engine_heuristic_api.cpp
    test_engine_api.cpp
    test_execution_plan_api.cpp
    test_get_error_api.cpp
    test_variant_pack_api.cpp
    test_handle_api.cpp
    test_util.cpp
)

target_compile_options(public_hipdnn_backend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(public_hipdnn_backend_tests
    GTest::gtest 
    GTest::gtest_main 
    Threads::Threads 
    hipdnn_sdk
    hipdnn_backend
    hip::host
    hip::device
)

target_include_directories(public_hipdnn_backend_tests
    PRIVATE
    ${HIPDNN_TEST_INCLUDE_DIRS}
)

target_compile_definitions(public_hipdnn_backend_tests
    PRIVATE
    TEST_GOOD_PLUGIN_NAME="${TEST_GOOD_PLUGIN_NAME}"
    TEST_EXECUTE_FAILS_PLUGIN_NAME="${TEST_EXECUTE_FAILS_PLUGIN_NAME}"
    TEST_NO_APPLICABLE_ENGINES_PLUGIN_NAME="${TEST_NO_APPLICABLE_ENGINES_PLUGIN_NAME}"
    TEST_DUPLICATE_ID_A_PLUGIN_NAME="${TEST_DUPLICATE_ID_A_PLUGIN_NAME}"
    TEST_DUPLICATE_ID_B_PLUGIN_NAME="${TEST_DUPLICATE_ID_B_PLUGIN_NAME}"
    TEST_INCOMPLETE_API_PLUGIN_NAME="${TEST_INCOMPLETE_API_PLUGIN_NAME}"
)

# Ensure test plugins are built before tests
add_dependencies(public_hipdnn_backend_tests test_good_plugin test_execute_fails_plugin test_no_applicable_engines_plugin test_good_default_plugin)

clang_tidy_check(public_hipdnn_backend_tests)
add_unit_test_target(public_hipdnn_backend_tests ${CMAKE_CURRENT_BINARY_DIR})
