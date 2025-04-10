# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
hipdnn_add_dependency(flatbuffers VERSION 23.1.21)

set(SCHEMA_FILES 
    schemas/batchnorm_inference_attributes.fbs
    schemas/data_types.fbs
    schemas/graph.fbs
    schemas/pointwise_attributes.fbs
    schemas/tensor_attributes.fbs )

_save_var(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS)

SET(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS "--gen-object-api;--gen-mutable;--gen-compare;--defaults-json")
build_flatbuffers("${SCHEMA_FILES}" "" generate_hipdnn_sdk_headers "" ${CMAKE_CURRENT_SOURCE_DIR}/include "" "") 

_restore_var(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS)