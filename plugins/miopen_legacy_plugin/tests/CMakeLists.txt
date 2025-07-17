# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(NOT BUILD_PLUGIN_AS_DEPENDENCY)
    find_package(GTest CONFIG REQUIRED)
endif()

include(GoogleTest)


if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping MIOpen Legacy Plugin tests")
    return()
endif()

add_executable(miopen_legacy_plugin_tests
    engines/test_miopen_engine.cpp
    engines/plans/test_miopen_batchnorm_plan_builder.cpp
    engines/plans/test_batchnorm_fwd_inference_plan.cpp
    main.cpp
    test_miopen_legacy_engine_plugin_api.cpp
    test_miopen_batchnorm_operations.cpp
    test_miopen_legacy_plugin_api.cpp
    test_miopen_handle_factory.cpp
    test_miopen_tensor.cpp
    test_miopen_utils.cpp
    test_engine_manager.cpp
    test_hipdnn_plugin_execution_context.cpp
)

target_compile_options(miopen_legacy_plugin_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(miopen_legacy_plugin_tests PUBLIC 
    GTest::gtest
    GTest::gmock
    miopen_legacy_plugin
    pthread
)

# make it so tests inside folders can find the mocks folder without having to use a relative path.
target_include_directories(miopen_legacy_plugin_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)


if(BUILD_PLUGIN_AS_DEPENDENCY)
    clang_tidy_check(miopen_legacy_plugin_tests)
    
    add_dependencies(check miopen_legacy_plugin_tests)
    gtest_discover_tests(
        miopen_legacy_plugin_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DISCOVERY_MODE PRE_TEST
    )
endif()