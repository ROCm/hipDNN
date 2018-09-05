# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

#Set default install and package paths.
set(CMAKE_INSTALL_PREFIX "/opt/rocm/" CACHE STRING "Install location.")
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm/hipdnn" CACHE STRING "Package install location.")


project(hipdnn CXX)
add_definitions(-std=c++11)
# HIP_PATH
IF(NOT DEFINED ENV{HIP_PATH})
  SET(HIP_PATH "/opt/rocm/hip")
ELSE()
  SET(HIP_PATH $ENV{HIP_PATH})
ENDIF()

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake  "${HIP_PATH}/cmake")
EXECUTE_PROCESS(COMMAND ${HIP_PATH}/bin/hipconfig -P OUTPUT_VARIABLE HIP_PLATFORM)

IF(${HIP_PLATFORM} MATCHES "hcc")
# MIOPEN
  if (NOT DEFINED ENV{MIOPEN_PATH})
    set(MIOPEN_PATH "/opt/rocm/miopen")
  else()
    set(MIOPEN_PATH $ENV{MIOPEN_PATH})
  endif()
ELSE()
  if (NOT DEFINED ENV{CUDNN_PATH})
    set(CUDNN_PATH "/usr/local/cudnn-7.0/cuda")
  else()
    set(CUDNN_PATH $ENV{CUDNN_PATH})
  endif()
ENDIF()

add_subdirectory(library)

#Get the current working branch
  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

# Get the latest abbreviated commit hash of the working branch
  execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(CPACK_PACKAGE_NAME "hipdnn")
  SET(CPACK_DEBIAN_PACKAGE_DESCRIPTION "hipDNN: a unified wrapper Library interface around cuDNN and MiOpen")
  set(CPACK_PACKAGE_VENDOR "MulticoreWare, Inc")
  set(CPACK_GENERATOR "DEB;TGZ" CACHE STRING "Packages generated.")
  set(CPACK_SOURCE_GENERATOR "TGZ")
  set(CPACK_BINARY_DEB "ON")
  set(CPACK_BINARY_STGZ "OFF")
  set(CPACK_SOURCE_TGZ "OFF")
  set(CPACK_SOURCE_TZ "OFF")
  set(CPACK_SOURCE_TBZ2 "OFF")
  set(CPACK_BINARY_TZ "OFF")
  SET(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${GIT_BRANCH}-${GIT_COMMIT_HASH}-${CMAKE_SYSTEM_NAME})
  SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Neelakandan <neelakandan@multicorewareinc.com>") #required

  # Debian specific flags
  set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
  "${CMAKE_SOURCE_DIR}/packaging/debian/postinst;${CMAKE_SOURCE_DIR}/packaging/debian/prerm")

  # RPM specific flags
  set(CPACK_RPM_PRE_INSTALL_SCRIPT_FILE "${CMAKE_SOURCE_DIR}/packaging/rpm/post")
  set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_SOURCE_DIR}/packaging/rpm/postun")

INCLUDE(CPack)

# Testing

enable_testing()

add_subdirectory(test)
