# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(NOT BUILD_PLUGIN_AS_DEPENDENCY)
    find_package(GTest CONFIG REQUIRED)

list(PREPEND HIPDNN_WARNING_COMPILE_OPTIONS
    -Werror                                  # Treat all warnings as errors
    -Wall                                    # Enable most common warnings
    -Wextra                                  # Enable additional warnings not covered by -Wall
    -Wpedantic                               # Enforce strict ISO C++ compliance
    -Wshadow                                 # Warn about variable shadowing
    -Wnon-virtual-dtor                       # Warn if a class with virtual functions has a non-virtual destructor
    -Wold-style-cast                         # Warn about C-style casts
    -Wcast-align                             # Warn about potential performance issues with misaligned casts
    -Woverloaded-virtual                     # Warn if a base class function is hidden by a derived class function with the same name
    -Wconversion                             # Warn about implicit type conversions that may alter a value
    -Wsign-conversion                        # Warn about implicit conversions between signed and unsigned types
    -Wnull-dereference                       # Warn about dereferencing null pointers
    -Wdouble-promotion                       # Warn when a float is implicitly promoted to a double
    -Wformat=2                               # Enable stricter format string checks
    -Winit-self                              # Warn about variables initialized with itself
    -Wunreachable-code                       # Warn about unreachable code
    -Wno-error=unused-command-line-argument  # Disable error for unused command line arguments
)
endif()

include(GoogleTest)


if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping MIOpen Legacy Plugin tests")
    return()
endif()

add_executable(miopen_legacy_plugin_tests
    engines/test_miopen_engine.cpp
    engines/solvers/test_miopen_batchnorm_solver.cpp
    main.cpp
    test_miopen_legacy_engine_plugin_api.cpp
    test_miopen_batchnorm_operations.cpp
    test_miopen_legacy_plugin_api.cpp
    test_miopen_handle_factory.cpp
    test_engine_manager.cpp
)

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

target_compile_options(miopen_legacy_plugin_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

if(BUILD_PLUGIN_AS_DEPENDENCY)
    clang_tidy_check(miopen_legacy_plugin_tests)
    
    add_dependencies(check miopen_legacy_plugin_tests)
    gtest_discover_tests(
        miopen_legacy_plugin_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DISCOVERY_MODE PRE_TEST
    )
endif()