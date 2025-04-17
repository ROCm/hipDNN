# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
find_package(hip REQUIRED)

add_library(hipdnn_backend_private
    descriptors/graph_descriptor.cpp
    descriptors/descriptor_factory.cpp
    flatbuffer_utilities.cpp
    plugin/engine_plugin.cpp
    plugin/plugin_core.cpp
)

target_include_directories(hipdnn_backend_private 
    PUBLIC 
        ${HIP_DNN_BACKEND_INCLUDE_DIR}
        ${HIP_DNN_BACKEND_SRC_DIR}
        ${HIP_DNN_SDK_INCLUDE_DIR}
        ${HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR}
)
target_link_libraries(hipdnn_backend_private PRIVATE 
    flatbuffers)

target_link_libraries(hipdnn_backend_private PUBLIC Boost::dll hip::device )

clang_tidy_check(hipdnn_backend_private)

add_library(hipdnn_backend SHARED 
    hipdnn_backend.cpp
)

target_include_directories(hipdnn_backend 
PUBLIC 
    ${HIP_INCLUDE_DIRS}
    ${HIP_DNN_BACKEND_INCLUDE_DIR} 
    ${HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR})

target_link_libraries(hipdnn_backend PRIVATE hipdnn_backend_private flatbuffers)
target_link_libraries(hipdnn_backend PRIVATE "-Xlinker --exclude-libs=ALL" ) # HIDE symbols from linked static libs

clang_tidy_check(hipdnn_backend)

include( GenerateExportHeader )
generate_export_header( hipdnn_backend 
                        EXPORT_FILE_NAME 
                        ${HIP_DNN_BACKEND_EXPORT_INCLUDE_DIR}/hipdnn_backend_export.h )