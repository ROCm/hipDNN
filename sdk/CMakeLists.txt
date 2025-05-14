# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

SET(HIP_DNN_SDK_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/hipdnn/sdk")

hipdnn_add_dependency(flatbuffers VERSION 23.1.21)

# Hack to workaround an issue with how flatbuffers sets up its INSTALL_INTERFACE.
# internally it uses the following: $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include>
# This end ups resolving to /opt/rocm/include/include which isnt correct.  I cant find a way
# to set this value when adding the dependency so after we init the dependency, I then force the
# INTERFACE_INCLUDE_DIRECTORIES to be the correct value required for the hipdnn_sdk
get_target_property(current_includes FlatBuffers INTERFACE_INCLUDE_DIRECTORIES)
list(REMOVE_ITEM current_includes "$<INSTALL_INTERFACE:include/include>")
list(APPEND current_includes "$<INSTALL_INTERFACE:${HIP_DNN_SDK_INSTALL_INCLUDE_DIR}/include>")
set_target_properties(FlatBuffers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${current_includes}"
)

set(HIP_DNN_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(SCHEMA_FILES 
    schemas/batchnorm_attributes.fbs
    schemas/batchnorm_backward_attributes.fbs
    schemas/batchnorm_inference_attributes.fbs
    schemas/data_types.fbs
    schemas/graph.fbs
    schemas/pointwise_attributes.fbs
    schemas/tensor_attributes.fbs )

_save_var(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS)

SET(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS "--gen-object-api;--gen-mutable;--gen-compare;--defaults-json")
build_flatbuffers(
    "${SCHEMA_FILES}" 
    "" 
    generate_hipdnn_sdk_headers 
    "" 
    ${HIP_DNN_SDK_INCLUDE_DIR}/hipdnn_sdk/data_objects 
    "" 
    ""
) 

# Flatc build has some warnings that trigger and print to the console.  Since we dont want these we can
# suppress all warnings by using the -w compile flag.  This command prepends -w to the compile flats for
# the flatc target.
set_target_properties(flatc PROPERTIES COMPILE_FLAGS "-w")


_restore_var(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS)

hipdnn_add_dependency(spdlog VERSION 1.15.2)

add_library(hipdnn_sdk INTERFACE)
target_link_libraries(hipdnn_sdk INTERFACE FlatBuffers spdlog_header_only)

target_include_directories(hipdnn_sdk INTERFACE
    $<BUILD_INTERFACE:${HIP_DNN_SDK_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${HIP_DNN_SDK_INSTALL_INCLUDE_DIR}/include>
)

install( 
    TARGETS hipdnn_sdk
    EXPORT hipdnn_sdk_targets
)

#todo, get spdlog to install its headers
install(
    DIRECTORY ${HIP_DNN_SDK_INCLUDE_DIR} ${HIP_DNN_FLATBUFFERS_INCLUDE_DIR} ${HIP_DNN_SPDLOG_INCLUDE_DIR}
    DESTINATION ${HIP_DNN_SDK_INSTALL_INCLUDE_DIR}
)

install(EXPORT hipdnn_sdk_targets
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/hipdnn_sdk
    FILE hipdnn_sdkConfig.cmake
)