# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping MIOpen Legacy Plugin tests")
    return()
endif()

add_executable(miopen_legacy_plugin_tests
    main.cpp
    miopen_legacy_plugin_basic_api_tests.cpp
)

target_link_libraries(miopen_legacy_plugin_tests PUBLIC 
    GTest::gtest
    miopen_legacy_plugin
    pthread
)

target_compile_options(miopen_legacy_plugin_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
clang_tidy_check(miopen_legacy_plugin_tests)

add_dependencies(check miopen_legacy_plugin_tests)
gtest_discover_tests(
    miopen_legacy_plugin_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISCOVERY_MODE PRE_TEST
)