# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(ENABLE_CLANG_TIDY)
    get_filename_component(COMPILER_PATH "${CMAKE_CXX_COMPILER}" PATH)
    find_program(CLANG_TIDY_EXE
        NAMES
            clang-tidy-20
            clang-tidy
        PATHS
            /usr/bin
            /opt/rocm/llvm/bin
            ${COMPILER_PATH}
    )

    function(find_clang_tidy_version VAR)
        execute_process(COMMAND ${CLANG_TIDY_EXE} -version OUTPUT_VARIABLE VERSION_OUTPUT)
        separate_arguments(VERSION_OUTPUT_LIST NATIVE_COMMAND "${VERSION_OUTPUT}")
        list(FIND VERSION_OUTPUT_LIST "version" VERSION_INDEX)
        if(VERSION_INDEX GREATER 0)
            math(EXPR VERSION_INDEX "${VERSION_INDEX} + 1")
            list(GET VERSION_OUTPUT_LIST ${VERSION_INDEX} VERSION)
            set(${VAR} ${VERSION} PARENT_SCOPE)
        else()
            set(${VAR} "0.0" PARENT_SCOPE)
        endif()
    endfunction()

    # Check clang-tidy version
    set(EXPECTED_CLANG_TIDY_VERSION "20")
    
    if(NOT CLANG_TIDY_EXE)
        message(STATUS "Clang tidy not found")
        set(CLANG_TIDY_VERSION "0.0")
    else()
        find_clang_tidy_version(CLANG_TIDY_VERSION)
        
        # Extract major version number
        if(CLANG_TIDY_VERSION MATCHES "^([0-9]+)\\.")
            set(CLANG_TIDY_MAJOR_VERSION "${CMAKE_MATCH_1}")
            if(NOT CLANG_TIDY_MAJOR_VERSION STREQUAL EXPECTED_CLANG_TIDY_VERSION)
                message(WARNING "clang-tidy version mismatch! Expected: ${EXPECTED_CLANG_TIDY_VERSION}, Found: ${CLANG_TIDY_MAJOR_VERSION}, Full version: ${CLANG_TIDY_VERSION}")
            else()
                message(STATUS "Found clang-tidy version ${CLANG_TIDY_MAJOR_VERSION} at ${CLANG_TIDY_EXE}")
            endif()
        else()
            message(WARNING "Could not determine clang-tidy major version from: ${CLANG_TIDY_VERSION}")
        endif()
    endif()

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    set(CLANG_TIDY_COMMAND 
        ${CLANG_TIDY_EXE}
        --config-file=${CMAKE_SOURCE_DIR}/.clang-tidy
        -p ${CMAKE_BINARY_DIR}
    )
endif()

function(clang_tidy_check TARGET)
    if(ENABLE_CLANG_TIDY)
        set_target_properties(${TARGET} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
    endif()
endfunction()
