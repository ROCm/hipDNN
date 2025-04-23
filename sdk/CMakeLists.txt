# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
hipdnn_add_dependency(flatbuffers VERSION 23.1.21)

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
build_flatbuffers("${SCHEMA_FILES}" "" generate_hipdnn_sdk_headers "" ${HIP_DNN_SDK_INCLUDE_DIR}/hipdnn_sdk/data_objects "" "") 

_restore_var(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS)

add_library(hipdnn_sdk INTERFACE)
target_link_libraries(hipdnn_sdk INTERFACE FlatBuffers)
target_compile_options(hipdnn_sdk INTERFACE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_include_directories(hipdnn_sdk INTERFACE
    $<BUILD_INTERFACE:${HIP_DNN_SDK_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

rocm_install_targets(
    TARGETS hipdnn_sdk
    INCLUDE ${HIP_DNN_SDK_INCLUDE_DIR} ${HIP_DNN_FLATBUFFERS_INCLUDE_DIR}
)