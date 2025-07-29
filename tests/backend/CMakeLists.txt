# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)
find_package(hip REQUIRED)

add_executable(public_hipdnn_backend_tests 
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
    pthread 
    hipdnn_sdk
    hipdnn_backend
    hipdnn_frontend
    hip::host
)

target_include_directories(public_hipdnn_backend_tests
    PRIVATE
    ${HIPDNN_TEST_INCLUDE_DIRS}
)

clang_tidy_check(public_hipdnn_backend_tests)
add_target_to_check_targets(public_hipdnn_backend_tests ${CMAKE_CURRENT_BINARY_DIR})
