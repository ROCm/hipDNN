# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

find_package(hip REQUIRED)
find_package(Threads REQUIRED)

# TODO Do not enable HIP language globally, use HIP RTC instead.
# The library does not contain any HIP code (or any other device code).
# It is required only for hipdnn_test_engine_plugin1.
enable_language(HIP)

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Backend Unit tests")
    return()
endif()

add_executable(hipdnn_backend_tests
    descriptors/TestBackendDescriptor.cpp
    descriptors/TestDescriptorFactory.cpp
    descriptors/TestEngineConfigDescriptor.cpp
    descriptors/TestEngineDescriptor.cpp
    descriptors/TestEngineHeuristicDescriptor.cpp
    descriptors/TestExecutionPlanDescriptor.cpp
    descriptors/TestGraphDescriptor.cpp
    descriptors/TestVariantPack.cpp
    main.cpp
    TestEnginePlugin.cpp
    TestEnginePluginResourceManager.cpp
    TestError.cpp
    TestFlatbufferUtilities.cpp
    TestHandle.cpp
    TestHelpers.cpp
    TestBackendLogger.cpp
    TestPlugin.cpp
    TestSharedLibrary.cpp
)

target_link_libraries(hipdnn_backend_tests 
    GTest::gtest
    GTest::gmock
    Threads::Threads
    hipdnn_backend_private
    hipdnn_sdk
    ${CMAKE_DL_LIBS}
)


target_compile_definitions(hipdnn_backend_tests PRIVATE HIPDNN_BACKEND_COMPILATION)
target_compile_options(hipdnn_backend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
clang_tidy_check(hipdnn_backend_tests)

# libhipdnn_test_plugin1.so or hipdnn_test_plugin1.dll (used by TestPlugin.cpp and TestSharedLibrary.cpp)
add_library(hipdnn_test_plugin1 SHARED plugins/PluginApiImpl.cpp plugins/Plugin1.cpp)
target_link_libraries(hipdnn_test_plugin1 PRIVATE hipdnn_sdk)
set_target_properties(hipdnn_test_plugin1 PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_options(hipdnn_test_plugin1 PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_compile_definitions(hipdnn_test_plugin1 PRIVATE COMPONENT_NAME="plugin1")
clang_tidy_check(hipdnn_test_plugin1)
add_dependencies(hipdnn_backend_tests hipdnn_test_plugin1)

# libhipdnn_test_plugin2.so or hipdnn_test_plugin2.dll (used by TestPlugin.cpp)
add_library(hipdnn_test_plugin2 SHARED plugins/PluginApiImpl.cpp plugins/Plugin2.cpp)
target_link_libraries(hipdnn_test_plugin2 PRIVATE hipdnn_sdk)
set_target_properties(hipdnn_test_plugin2 PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_options(hipdnn_test_plugin2 PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_compile_definitions(hipdnn_test_plugin2 PRIVATE COMPONENT_NAME="plugin2")
clang_tidy_check(hipdnn_test_plugin2)
add_dependencies(hipdnn_backend_tests hipdnn_test_plugin2)

# libhipdnn_test_engine_plugin1.so or hipdnn_test_engine_plugin1.dll (used by TestEnginePlugin.cpp)
add_library(hipdnn_test_engine_plugin1 SHARED plugins/PluginApiImpl.cpp plugins/EnginePluginApiImpl.cpp plugins/EnginePlugin1.cpp)
# TODO Use HIP RTC
set_source_files_properties(plugins/EnginePlugin1.cpp PROPERTIES LANGUAGE HIP)
if (DEFINED GPU_TARGETS)
    set_property(TARGET hipdnn_test_engine_plugin1 PROPERTY HIP_ARCHITECTURES ${GPU_TARGETS})
else()
    get_property(GPU_TARGETS GLOBAL PROPERTY HIP_ARCHITECTURES)
endif()
message(STATUS "Architectures for hipdnn_test_engine_plugin1: ${GPU_TARGETS}")
target_link_libraries(hipdnn_test_engine_plugin1 PRIVATE hipdnn_sdk hip::host)
set_target_properties(hipdnn_test_engine_plugin1 PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_options(hipdnn_test_engine_plugin1 PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

clang_tidy_check(hipdnn_test_engine_plugin1)
add_dependencies(hipdnn_backend_tests hipdnn_test_engine_plugin1)

add_unit_test_target(hipdnn_backend_tests ${CMAKE_CURRENT_BINARY_DIR})
