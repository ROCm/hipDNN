# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21...3.22)

include(FetchContent)

option(HIPDNN_NO_DOWNLOAD
       "Disables downloading of any external dependencies" OFF
)

if(HIPDNN_NO_DOWNLOAD)
    set(FETCHCONTENT_FULLY_DISCONNECTED
        OFF
        CACHE BOOL "Don't attempt to download or update anything" FORCE
    )
endif()

# Dependencies where the local version should be used, if available
set(_hipdnn_all_local_deps
    ROCmCMakeBuildTools
)
# Dependencies where we never look for a local version
set(_hipdnn_all_remote_deps
    boost
    googletest
    flatbuffers
)

# hipdnn_add_dependency(
#   dep_name
#   [NO_LOCAL]
#   [VERSION version]
#   [FIND_PACKAGE_ARGS args...]
#   [COMPONENT component]
#   [PACKAGE_NAME
#      [package_name]
#      [DEB deb_package_name]
#      [RPM rpm_package_name]]
# )
function(hipdnn_add_dependency dep_name)
    set(options NO_LOCAL)
    set(oneValueArgs VERSION HASH)
    set(multiValueArgs FIND_PACKAGE_ARGS PACKAGE_NAME COMPONENTS)
    cmake_parse_arguments(
        PARSE
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    if(dep_name IN_LIST _hipdnn_all_local_deps)
        if(NOT PARSE_NO_LOCAL)
            find_package(
                ${dep_name} ${PARSE_VERSION} QUIET ${PARSE_FIND_PACKAGE_ARGS}
            )
        endif()
        if(NOT ${dep_name}_FOUND)
            message(STATUS "Did not find ${dep_name}, it will be built locally")
            _build_local()
        else()
            message(
                STATUS
                    "Found ${dep_name}: ${${dep_name}_DIR} (found version \"${${dependency_name}_VERSION}\")"
            )
            foreach(VAR IN LISTS ${dep_name}_EXPORT_VARS)
                set(${VAR}
                    ${${VAR}}
                    PARENT_SCOPE
                )
            endforeach()
        endif()
    elseif(dep_name IN_LIST _hipdnn_all_remote_deps)
        message(TRACE "Will build ${dep_name} locally")
        _build_local()
    else()
        message(WARNING "Unknown dependency: ${dep_name}")
        return()
    endif()

    if(PARSE_COMPONENTS AND PROJECT_IS_TOP_LEVEL)
        if(COMMAND rocm_package_add_dependencies)
            unset(name_deb)
            unset(name_rpm)
            if(PARSE_PACKAGE_NAME)
                cmake_parse_arguments(
                    PKG_NAME
                    ""
                    "DEB;RPM"
                    ""
                    ${PARSE_PACKAGE_NAME}
                )
                if(PKG_NAME_DEB)
                    set(name_deb "${PKG_NAME_DEB}")
                elseif(PKG_NAME_UNPARSED_ARGUMENTS)
                    set(name_deb "${PKG_NAME_UNPARSED_ARGUMENTS}")
                endif()
                if(PKG_NAME_RPM)
                    set(name_rpm "${PKG_NAME_RPM}")
                elseif(PKG_NAME_UNPARSED_ARGUMENTS)
                    set(name_rpm "${PKG_NAME_UNPARSED_ARGUMENTS}")
                endif()
            endif()
            foreach(COMPONENT IN LISTS PARSE_COMPONENTS)
                if("${name_deb}" STREQUAL "")
                    set(name_deb "lib${dep_name}-dev")
                endif()
                if("${name_rpm}" STREQUAL "")
                    set(name_rpm "lib${dep_name}-devel")
                endif()
                if(PARSE_VERSION)
                    set(VERSION_STR " >= ${PARSE_VERSION}")
                endif()
                rocm_package_add_deb_dependencies(
                    QUIET
                    COMPONENT ${COMPONENT}
                    DEPENDS "${name_deb}${VERSION_STR}"
                )
                rocm_package_add_rpm_dependencies(
                    QUIET
                    COMPONENT ${COMPONENT}
                    DEPENDS "${name_rpm}${VERSION_STR}"
                )
                string(TOUPPER "${COMPONENT}" COMPONENT_VAR)
                set(CPACK_DEBIAN_${COMPONENT_VAR}_PACKAGE_DEPENDS
                    "${CPACK_DEBIAN_${COMPONENT_VAR}_PACKAGE_DEPENDS}"
                    PARENT_SCOPE
                )
                set(CPACK_RPM_${COMPONENT_VAR}_PACKAGE_REQUIRES
                    "${CPACK_RPM_${COMPONENT_VAR}_PACKAGE_REQUIRES}"
                    PARENT_SCOPE
                )
            endforeach()
        else()
            message(
                ERROR
                "ROCmCMakeBuildTools is required to add dependencies to a package"
            )
        endif()
    endif()
endfunction()

macro(_build_local)
    cmake_policy(PUSH)
    if(BUILD_VERBOSE)
        message(STATUS "=========== Adding ${dep_name} ===========")
    endif()
    _pushstate()
    set(CMAKE_MESSAGE_INDENT "[${dep_name}] ")
    cmake_language(
        CALL _fetch_${dep_name} "${PARSE_VERSION}" "${PARSE_HASH}"
    )
    _popstate()
    if(BUILD_VERBOSE)
        message(STATUS "=========== Added ${dep_name} ===========")
    endif()
    cmake_policy(POP)
    foreach(VAR IN LISTS ${dep_name}_EXPORT_VARS)
        set(${VAR}
            ${${VAR}}
            PARENT_SCOPE
        )
    endforeach()
endmacro()

function(_fetch_boost VERSION HASH)
    _determine_git_tag("boost-" "boost-1.81.0")
    FetchContent_Declare(
        boost
        URL https://github.com/boostorg/boost/releases/download/${GIT_TAG}/${GIT_TAG}.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    _save_var(BUILD_TESTING)
    set(BUILD_TESTING OFF)
    _save_var(BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_MakeAvailable(boost)
    _restore_var(BUILD_SHARED_LIBS)
    _restore_var(BUILD_TESTING)
    _exclude_from_all(${boost_SOURCE_DIR})
    _mark_targets_as_system(${boost_SOURCE_DIR})
endfunction()

function(_fetch_googletest VERSION HASH)
    if(VERSION AND VERSION STREQUAL 1.12.1)
        set(GIT_TAG release-1.12.1)
    else()
        _determine_git_tag(v release-1.12.1)
    endif()
    if(HASH)
        set(HASH_ARG HASH ${HASH})
    endif()
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/${GIT_TAG}.zip
        ${HASH_ARG}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    option(HIPDNN_GTEST_SHARED "Build GTest as a shared library." OFF)
    _save_var(BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS
        ${HIPDNN_GTEST_SHARED}
        CACHE INTERNAL ""
    )
    set(INSTALL_GTEST OFF)
    FetchContent_MakeAvailable(googletest)
    _restore_var(BUILD_SHARED_LIBS)

    _exclude_from_all(${googletest_SOURCE_DIR})
    _mark_targets_as_system(${googletest_SOURCE_DIR})
    target_compile_options(gtest PUBLIC ${EXTRA_COMPILE_OPTIONS})
    target_compile_options(gmock PUBLIC ${EXTRA_COMPILE_OPTIONS})
endfunction()

function(_fetch_flatbuffers VERSION HASH)
    _determine_git_tag(v 23.1.21)
    
    _save_var(FLATBUFFERS_BUILD_FLATC)
    _save_var(FLATBUFFERS_INSTALL)
    _save_var(FLATBUFFERS_BUILD_TESTS)
    _save_var(FLATBUFFERS_BUILD_FLATHASH)
    _save_var(FLATBUFFERS_ENABLE_PCH)

    set(FLATBUFFERS_BUILD_FLATC ON)
    set(FLATBUFFERS_INSTALL OFF)
    set(FLATBUFFERS_BUILD_TESTS OFF)
    set(FLATBUFFERS_BUILD_FLATHASH OFF)
    set(FLATBUFFERS_ENABLE_PCH ON)

    FetchContent_Declare(
        flatbuffers
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        GIT_TAG ${GIT_TAG}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    FetchContent_MakeAvailable(flatbuffers)

    _restore_var(FLATBUFFERS_BUILD_FLATC)
    _restore_var(FLATBUFFERS_INSTALL)
    _restore_var(FLATBUFFERS_BUILD_TESTS)
    _restore_var(FLATBUFFERS_BUILD_FLATHASH)
    _restore_var(FLATBUFFERS_ENABLE_PCH)
    
    _exclude_from_all(${flatbuffers_SOURCE_DIR})
    _mark_targets_as_system(${flatbuffers_SOURCE_DIR})
endfunction()

function(_fetch_ROCmCMakeBuildTools VERSION HASH)
    _determine_git_tag(FALSE rocm-6.3.0)
    FetchContent_Declare(
        ROCmCMakeBuildTools
        GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
        GIT_TAG ${GIT_TAG}
        SOURCE_SUBDIR "DISABLE ADDING TO BUILD"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        # Don't consume the build/test targets of ROCmCMakeBuildTools
    )
    FetchContent_MakeAvailable(ROCmCMakeBuildTools)
    list(
        APPEND CMAKE_MODULE_PATH
        ${rocmcmakebuildtools_SOURCE_DIR}/share/rocmcmakebuildtools/cmake
    )
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} PARENT_SCOPE)
endfunction()
set(ROCmCMakeBuildTools_EXPORT_VARS CMAKE_MODULE_PATH)

# Utility functions, pulled from rocroller repo
macro(_determine_git_tag PREFIX DEFAULT)
    if(HASH)
        set(GIT_TAG ${HASH})
    elseif(VERSION AND NOT "${PREFIX}" STREQUAL "FALSE")
        set(GIT_TAG ${PREFIX}${VERSION})
    else()
        set(GIT_TAG ${DEFAULT})
    endif()
endmacro()

macro(_save_var _name)
    if(DEFINED CACHE{${_name}})
        set(_old_cache_${_name} $CACHE{${_name}})
        unset(${_name} CACHE)
    endif()
    # We can't tell if a variable is referring to a cache or a regular variable.
    # To ensure this gets the value of the regular, variable, temporarily unset
    # the cache variable if it was set before checking for the regular variable.
    if(DEFINED ${_name})
        set(_old_${_name} ${${_name}})
    endif()
    if(DEFINED _old_cache_${_name})
        set(${_name}
            ${_old_cache_${_name}}
            CACHE INTERNAL ""
        )
    endif()
    if(DEFINED ENV{${_name}})
        set(_old_env_${_name} $ENV{${_name}})
    endif()
endmacro()

macro(_restore_var _name)
    if(DEFINED _old_${_name})
        set(${_name} ${_old_${_name}})
        unset(_old_${_name})
    else()
        unset(${_name})
    endif()
    if(DEFINED _old_cache_${_name})
        set(${_name}
            ${_old_cache_${_name}}
            CACHE INTERNAL ""
        )
        unset(_old_cache_${_name})
    else()
        unset(${_name} CACHE)
    endif()
    if(DEFINED _old_env_${_name})
        set(ENV{${_name}} ${_old_env_${_name}})
        unset(_old_env_${_name})
    else()
        unset(ENV{${_name}})
    endif()
endmacro()

## not actually a stack, but that shouldn't be relevant
macro(_pushstate)
    _save_var(CMAKE_CXX_CPPCHECK)
    unset(CMAKE_CXX_CPPCHECK)
    unset(CMAKE_CXX_CPPCHECK CACHE)
    _save_var(CMAKE_MESSAGE_INDENT)
    _save_var(CPACK_GENERATOR)
endmacro()

macro(_popstate)
    _restore_var(CPACK_GENERATOR)
    _restore_var(CMAKE_MESSAGE_INDENT)
    _restore_var(CMAKE_CXX_CPPCHECK)
endmacro()

macro(_exclude_from_all _dir)
    set_property(DIRECTORY ${_dir} PROPERTY EXCLUDE_FROM_ALL ON)
endmacro()

macro(_mark_targets_as_system _dirs)
    foreach(_dir ${_dirs})
        get_directory_property(_targets DIRECTORY ${_dir} BUILDSYSTEM_TARGETS)
        foreach(_target IN LISTS _targets)
            get_target_property(
                _includes ${_target} INTERFACE_INCLUDE_DIRECTORIES
            )
            if(_includes)
                target_include_directories(
                    ${_target} SYSTEM INTERFACE ${_includes}
                )
            endif()
        endforeach()
    endforeach()
endmacro()
