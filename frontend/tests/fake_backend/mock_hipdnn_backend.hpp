// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <cstdint>
#include <gmock/gmock.h>

#define HIPDNN_BACKEND_STATIC_DEFINE

#include "hipdnn_backend.h"

// NOLINTBEGIN

class Mock_hipdnn_backend : public hipdnn_frontend::Hipdnn_backend_interface
{
public:
    MOCK_METHOD(hipdnnStatus_t, create, (hipdnnHandle_t * handle), ());
    MOCK_METHOD(hipdnnStatus_t, destroy, (hipdnnHandle_t handle), ());
    MOCK_METHOD(hipdnnStatus_t, set_stream, (hipdnnHandle_t handle, hipStream_t streamId), ());
    MOCK_METHOD(hipdnnStatus_t, get_stream, (hipdnnHandle_t handle, hipStream_t* streamId), ());
    MOCK_METHOD(hipdnnStatus_t,
                backend_create_descriptor,
                (hipdnnBackendDescriptorType_t descriptor_type,
                 hipdnnBackendDescriptor_t* descriptor),
                ());
    MOCK_METHOD(hipdnnStatus_t,
                backend_destroy_descriptor,
                (hipdnnBackendDescriptor_t descriptor),
                ());
    MOCK_METHOD(hipdnnStatus_t,
                backend_execute,
                (hipdnnHandle_t handle,
                 hipdnnBackendDescriptor_t execution_plan,
                 hipdnnBackendDescriptor_t variant_pack),
                ());
    MOCK_METHOD(hipdnnStatus_t, backend_finalize, (hipdnnBackendDescriptor_t descriptor), ());
    MOCK_METHOD(hipdnnStatus_t,
                backend_get_attribute,
                (hipdnnBackendDescriptor_t descriptor,
                 hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t requested_element_count,
                 int64_t* element_count,
                 void* array_of_elements),
                ());
    MOCK_METHOD(hipdnnStatus_t,
                backend_set_attribute,
                (hipdnnBackendDescriptor_t descriptor,
                 hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t element_count,
                 const void* array_of_elements),
                ());
    MOCK_METHOD(const char*, get_error_string, (hipdnnStatus_t status), ());
    MOCK_METHOD(void, get_last_error_string, (char* message, size_t max_size), ());
    MOCK_METHOD(hipdnnStatus_t,
                backend_create_and_deserialize_graph_ext,
                (hipdnnBackendDescriptor_t * descriptor,
                 const uint8_t* serialized_graph,
                 size_t graph_byte_size),
                ());
    MOCK_METHOD(void, logging_callback_ext, (hipdnnSeverity_t severity, const char* msg), ());
    MOCK_METHOD(hipdnnStatus_t,
                set_engine_plugin_paths_ext,
                (size_t num_paths,
                 const char* const* plugin_paths,
                 hipdnnPluginLoadingMode_ext_t mode),
                ());
};

// NOLINTEND
