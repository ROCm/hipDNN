CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8)
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${HIP_PATH}/cmake ${MIOPEN_PATH}/lib/cmake/miopen)

execute_process(COMMAND ${HIP_PATH}/bin/hipconfig --platform OUTPUT_VARIABLE HIP_PLATFORM)
MESSAGE (STATUS "HIP_PATH : ${HIP_PATH}")

#Make sure HIP is installed in the target system
FIND_PACKAGE(HIP 1.0 REQUIRED)
add_subdirectory(./utils/csv)

SET(CMAKE_CXX_COMPILER "${HIP_PATH}/bin/hipcc")
if(${HIP_PLATFORM} MATCHES "nvcc")
    set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "-Xcompiler ${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS}")
endif()
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/library/include   ${HIP_PATH}/include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/utils/)
FILE(GLOB HIPDNNTESTSRCS "${CMAKE_CURRENT_SOURCE_DIR}/utils/src/*.cc" ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
INCLUDE_DIRECTORIES(${MIOPEN_PATH}/include/)
ADD_EXECUTABLE(unittest ${HIPDNNTESTSRCS})

TARGET_LINK_LIBRARIES(unittest csv_integration)
if(${HIP_PLATFORM} MATCHES "nvcc")
    set_target_properties(unittest PROPERTIES LINK_FLAGS "-Xcompiler \\\"-Wl\\\\,-rpath=./library/\\\"")
endif()

set_target_properties(unittest PROPERTIES SKIP_BUILD_RPATH 1)
set_target_properties(unittest PROPERTIES LINK_FLAGS "${CMAKE_SOURCE_DIR}/build/library/libhipdnn.so")
FILE(GLOB HIPDNNRESULT "${CMAKE_CURRENT_SOURCE_DIR}/utils/csv/compare_result.cpp")
ADD_EXECUTABLE(compare_results ${HIPDNNRESULT})
file(COPY ${CMAKE_SOURCE_DIR}/execute.sh
     DESTINATION ${CMAKE_SOURCE_DIR}/build)