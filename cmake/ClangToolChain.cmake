# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Platform-specific compiler configuration

if(UNIX)
    # Unix/Linux: Use ROCm LLVM Clang
    set(ROCM_LLVM_BIN_DIR /opt/rocm/llvm/bin)
    set(ROCM_LLVM_LIB_DIR /opt/rocm/llvm/lib)

    if(EXISTS ${ROCM_LLVM_BIN_DIR})
        # Set the C and C++ compilers to clang and clang++ with a specific directory hint
        set(CMAKE_C_COMPILER ${ROCM_LLVM_BIN_DIR}/clang)
        set(CMAKE_CXX_COMPILER ${ROCM_LLVM_BIN_DIR}/clang++)
        set(CMAKE_SYMBOLIZER ${ROCM_LLVM_BIN_DIR}/llvm-symbolizer)
        message(STATUS "Using ROCm Clang compilers from ${ROCM_LLVM_BIN_DIR}")
    else()
        message(WARNING "The directory /opt/rocm/llvm/bin does not exist. Cannot auto select clang compilers.")
    endif()

    add_compile_options(-fPIC) # Position Independent Code (not needed/supported on Windows)

elseif(WIN32)
    # Windows: Use Clang from TheRock build
    set(WINDOWS_ROCM_DIR "C:/src/TheRock/build/dist/rocm" CACHE PATH "Path to Windows ROCm installation")
    set(WINDOWS_ROCM_LLVM_BIN_DIR "${WINDOWS_ROCM_DIR}/lib/llvm/bin")
    set(WINDOWS_ROCM_CMAKE_DIR "${WINDOWS_ROCM_DIR}/lib/cmake")

    if(EXISTS ${WINDOWS_ROCM_LLVM_BIN_DIR})
        # Set the C and C++ compilers to clang and clang++ for Windows
        set(CMAKE_RC_COMPILER rc.exe)
        set(CMAKE_C_COMPILER ${WINDOWS_ROCM_LLVM_BIN_DIR}/clang.exe)
        set(CMAKE_CXX_COMPILER ${WINDOWS_ROCM_LLVM_BIN_DIR}/clang++.exe)
        set(CMAKE_HIP_COMPILER ${WINDOWS_ROCM_LLVM_BIN_DIR}/clang++.exe)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fms-extensions -fms-compatibility -fdeclspec")
        # set(CMAKE_HIP_COMPILER_ROCM_ROOT ${WINDOWS_ROCM_DIR})
        # set(CMAKE_HIP_ARCHITECTURES gfx1102)
        message(STATUS "Using Windows ROCm Clang compilers from ${WINDOWS_ROCM_LLVM_BIN_DIR}")
    else()
        message(WARNING "The directory ${WINDOWS_ROCM_LLVM_BIN_DIR} does not exist. Cannot auto select clang compilers.")
    endif()

    # Set up CMake package search path for TheRock build
    if(EXISTS ${WINDOWS_ROCM_CMAKE_DIR})
        # Add to CMAKE_PREFIX_PATH for find_package() searches
        list(APPEND CMAKE_PREFIX_PATH ${WINDOWS_ROCM_CMAKE_DIR})
        message(STATUS "Added Windows ROCm CMake package search path: ${WINDOWS_ROCM_CMAKE_DIR}")
    else()
        message(WARNING "Windows ROCm CMake directory not found: ${WINDOWS_ROCM_CMAKE_DIR}")
    endif()

    # Windows SDK Configuration
    # Allow user to specify Windows SDK path and version
    set(WINDOWS_SDK_PATH "" CACHE PATH "Path to Windows SDK installation (e.g., C:/Program Files (x86)/Windows Kits/10)")
    set(WINDOWS_SDK_VERSION "" CACHE STRING "Windows SDK version to use (e.g., 10.0.19041.0)")

    # Try to auto-detect Windows SDK if not specified
    if(NOT WINDOWS_SDK_PATH)
        # Common Windows SDK locations
        set(_sdk_search_paths
            "C:/Program Files (x86)/Windows Kits/10"
            "C:/Program Files/Windows Kits/10"
            "$ENV{ProgramFiles\(x86\)}/Windows Kits/10"
            "$ENV{ProgramFiles}/Windows Kits/10"
        )
        
        foreach(_path ${_sdk_search_paths})
            if(EXISTS "${_path}/Include")
                set(WINDOWS_SDK_PATH "${_path}")
                message(STATUS "Auto-detected Windows SDK at: ${WINDOWS_SDK_PATH}")
                break()
            endif()
        endforeach()
    endif()

    if(WINDOWS_SDK_PATH)
        # Auto-detect SDK version if not specified
        if(NOT WINDOWS_SDK_VERSION)
            file(GLOB _sdk_versions RELATIVE "${WINDOWS_SDK_PATH}/Include" "${WINDOWS_SDK_PATH}/Include/*")
            # Filter to only version-like directories (e.g., 10.0.xxxxx.x)
            set(_valid_versions)
            foreach(_version ${_sdk_versions})
                if(_version MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$")
                    list(APPEND _valid_versions ${_version})
                endif()
            endforeach()
            
            if(_valid_versions)
                # Sort and pick the latest version
                list(SORT _valid_versions COMPARE NATURAL ORDER DESCENDING)
                list(GET _valid_versions 0 WINDOWS_SDK_VERSION)
                message(STATUS "Auto-detected Windows SDK version: ${WINDOWS_SDK_VERSION}")
            endif()
        endif()

        if(WINDOWS_SDK_VERSION)
            # Determine target architecture
            if(CMAKE_GENERATOR_PLATFORM)
                set(_target_arch ${CMAKE_GENERATOR_PLATFORM})
            elseif(CMAKE_SYSTEM_PROCESSOR)
                set(_target_arch ${CMAKE_SYSTEM_PROCESSOR})
            else()
                set(_target_arch "x64")  # Default to x64
            endif()

            # Normalize architecture name
            if(_target_arch MATCHES "AMD64|x86_64|X64")
                set(_target_arch "x64")
            elseif(_target_arch MATCHES "X86|i386|i686")
                set(_target_arch "x86")
            elseif(_target_arch MATCHES "ARM64|AARCH64")
                set(_target_arch "arm64")
            elseif(_target_arch MATCHES "ARM")
                set(_target_arch "arm")
            endif()

            # Set up Windows SDK include directories
            set(WINDOWS_SDK_INCLUDE_DIRS
                "${WINDOWS_SDK_PATH}/Include/${WINDOWS_SDK_VERSION}/ucrt"
                "${WINDOWS_SDK_PATH}/Include/${WINDOWS_SDK_VERSION}/shared"
                "${WINDOWS_SDK_PATH}/Include/${WINDOWS_SDK_VERSION}/um"
                "${WINDOWS_SDK_PATH}/Include/${WINDOWS_SDK_VERSION}/winrt"
                "${WINDOWS_SDK_PATH}/Include/${WINDOWS_SDK_VERSION}/cppwinrt"
            )

            # Set up Windows SDK library directories
            set(WINDOWS_SDK_LIB_DIRS
                "${WINDOWS_SDK_PATH}/Lib/${WINDOWS_SDK_VERSION}/ucrt/${_target_arch}"
                "${WINDOWS_SDK_PATH}/Lib/${WINDOWS_SDK_VERSION}/um/${_target_arch}"
            )

            # Verify that the directories exist
            set(_sdk_valid TRUE)
            foreach(_dir ${WINDOWS_SDK_INCLUDE_DIRS})
                if(NOT EXISTS "${_dir}")
                    message(WARNING "Windows SDK include directory not found: ${_dir}")
                    set(_sdk_valid FALSE)
                endif()
            endforeach()

            foreach(_dir ${WINDOWS_SDK_LIB_DIRS})
                if(NOT EXISTS "${_dir}")
                    message(WARNING "Windows SDK library directory not found: ${_dir}")
                    set(_sdk_valid FALSE)
                endif()
            endforeach()

            if(_sdk_valid)
                # Add to CMake search paths
                list(APPEND CMAKE_INCLUDE_PATH ${WINDOWS_SDK_INCLUDE_DIRS})
                list(APPEND CMAKE_LIBRARY_PATH ${WINDOWS_SDK_LIB_DIRS})
                
                # Export for use in the project
                set(WINDOWS_SDK_INCLUDE_DIRS ${WINDOWS_SDK_INCLUDE_DIRS} CACHE INTERNAL "Windows SDK include directories")
                set(WINDOWS_SDK_LIB_DIRS ${WINDOWS_SDK_LIB_DIRS} CACHE INTERNAL "Windows SDK library directories")
                
                message(STATUS "Windows SDK configured:")
                message(STATUS "  Path: ${WINDOWS_SDK_PATH}")
                message(STATUS "  Version: ${WINDOWS_SDK_VERSION}")
                message(STATUS "  Architecture: ${_target_arch}")
                # message(STATUS "  Include dirs: ${WINDOWS_SDK_INCLUDE_DIRS}")
                # message(STATUS "  Library dirs: ${WINDOWS_SDK_LIB_DIRS}")
            else()
                message(WARNING "Windows SDK configuration is incomplete. Some directories are missing.")
            endif()
        else()
            message(WARNING "Windows SDK path found but no valid version detected.")
        endif()
    else()
        message(STATUS "Windows SDK not configured. Set WINDOWS_SDK_PATH to enable Windows SDK support.")
    endif()

    #add_compile_options(-fms-extensions)
endif()
