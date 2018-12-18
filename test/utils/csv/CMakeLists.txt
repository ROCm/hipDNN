cmake_minimum_required(VERSION 2.8.8)

IF(NOT DEFINED ENV{HIP_PATH})
  SET(HIP_PATH "/opt/rocm/hip")
ELSE()
  SET(HIP_PATH $ENV{HIP_PATH})
ENDIF()

SET(CMAKE_CXX_COMPILER "${HIP_PATH}/bin/hipcc")

include_directories( ${CMAKE_CURRENT_SOURCE_DIR})
EXECUTE_PROCESS(COMMAND ${HIP_PATH}/bin/hipconfig -P OUTPUT_VARIABLE HIP_PLATFORM)
include_directories(${HIP_PATH}/include/)
add_library(csv_integration STATIC csv_integration.cpp ${HEADER_FILES})