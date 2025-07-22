// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <cstdint>
#include <gmock/gmock.h>

#include "hipdnn_backend.h"

// NOLINTBEGIN

class Mock_hipdnn_backend
{
public:
    MOCK_METHOD(hipdnnStatus_t, hipdnnCreate, (hipdnnHandle_t * handle), ());
    MOCK_METHOD(hipdnnStatus_t, hipdnnDestroy, (hipdnnHandle_t handle), ());
    MOCK_METHOD(hipdnnStatus_t, hipdnnSetStream, (hipdnnHandle_t handle, hipStream_t streamId), ());
    MOCK_METHOD(hipdnnStatus_t,
                hipdnnGetStream,
                (hipdnnHandle_t handle, hipStream_t* streamId),
                ());
    MOCK_METHOD(hipdnnStatus_t,
                hipdnnBackendCreateDescriptor,
                (hipdnnBackendDescriptorType_t descriptor_type,
                 hipdnnBackendDescriptor_t* descriptor),
                ());
    MOCK_METHOD(hipdnnStatus_t,
                hipdnnBackendDestroyDescriptor,
                (hipdnnBackendDescriptor_t descriptor),
                ());
    MOCK_METHOD(hipdnnStatus_t,
                hipdnnBackendExecute,
                (hipdnnHandle_t handle,
                 hipdnnBackendDescriptor_t execution_plan,
                 hipdnnBackendDescriptor_t variant_pack),
                ());
    MOCK_METHOD(hipdnnStatus_t, hipdnnBackendFinalize, (hipdnnBackendDescriptor_t descriptor), ());
    MOCK_METHOD(hipdnnStatus_t,
                hipdnnBackendGetAttribute,
                (hipdnnBackendDescriptor_t descriptor,
                 hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t requested_element_count,
                 int64_t* element_count,
                 void* array_of_elements),
                ());
    MOCK_METHOD(hipdnnStatus_t,
                hipdnnBackendSetAttribute,
                (hipdnnBackendDescriptor_t descriptor,
                 hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t element_count,
                 const void* array_of_elements),
                ());
    MOCK_METHOD(const char*, hipdnnGetErrorString, (hipdnnStatus_t status), ());
    MOCK_METHOD(void, hipdnnGetLastErrorString, (char* message, size_t max_size), ());
    MOCK_METHOD(hipdnnStatus_t,
                hipdnnBackendCreateAndDeserializeGraph_ext,
                (hipdnnBackendDescriptor_t * descriptor,
                 const uint8_t* serialized_graph,
                 size_t graph_byte_size),
                ());
};

// NOLINTEND
