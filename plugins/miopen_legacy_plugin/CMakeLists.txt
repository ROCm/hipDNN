# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(NOT BUILD_PLUGIN_AS_DEPENDENCY)
    message(STATUS "Building miopen legacy plugin standalone")
    cmake_minimum_required(VERSION 3.25.2)
    add_compile_definitions(__HIP_PLATFORM_AMD__)
    
    if(NOT WIN32)
        set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")
    endif()

    set(MIOPEN_LEGACY_PLUGIN "0.0.1")
    project(
        hipDNN
        VERSION ${MIOPEN_LEGACY_PLUGIN}
        LANGUAGES CXX)
        
    set(CMAKE_CXX_STANDARD 20)

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
    
    # NOTE: Workaround until llvm & hip cmake modules fixes symlink logic in their config files; remove when fixed.
    #       Added to prevent issues with not finding hip cmake.
    list(
        APPEND
        CMAKE_PREFIX_PATH
        ${ROCM_PATH}/llvm
        ${ROCM_PATH}
        ${ROCM_PATH}/hip
        /opt/rocm/llvm
        /opt/rocm
        /opt/rocm/hip 
    )
    
    find_package(hip REQUIRED)
    find_package(hipdnn_sdk CONFIG REQUIRED)
else()
    message(STATUS "Building miopen legacy plugin as a dependency")
endif()

find_package(miopen CONFIG REQUIRED)

add_library(miopen_legacy_plugin SHARED
    engines/miopen_engine.cpp
    engines/plans/miopen_batchnorm_plan_builder.cpp
    engines/plans/miopen_batchnorm_fwd_inference_plan.cpp
    engine_manager.cpp
    miopen_legacy_plugin.cpp
    miopen_handle_factory.cpp
    miopen_container.cpp
    miopen_utils.cpp
    miopen_tensor.cpp
)

target_include_directories(miopen_legacy_plugin 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(miopen_legacy_plugin PUBLIC MIOpen hipdnn_sdk)
target_compile_options(miopen_legacy_plugin PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

add_subdirectory(tests)

install(
    TARGETS miopen_legacy_plugin
    EXPORT miopen-legacy-plugin
    LIBRARY DESTINATION lib/hipdnn_plugins
)
