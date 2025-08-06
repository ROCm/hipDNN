// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <tuple>

#include "fake_hipdnn_backend.hpp"
#include "hipdnn_backend.h"
#include "mock_hipdnn_backend.hpp"

// NOLINTBEGIN

namespace fake_hipdnn_backend
{

// We store a weak pointer here so that every test that needs a Mock_hipdnn_backend has to
// create its own copy of one.  Once a test finishes, its shared ptr should go out of scope
// and invalidate the weak pointer.  Not doing this could end up having a mock set here that
// has expectations/returns which could cause side effects in other tests.
std::weak_ptr<Mock_hipdnn_backend> mock_hipdnn_backend;

std::shared_ptr<Mock_hipdnn_backend> backend()
{
    auto shared_backend = mock_hipdnn_backend.lock();
    if(!shared_backend)
    {
        throw std::runtime_error("Mock hipDNN backend is not set. Please set it using "
                                 "set_mock_hipdnn_backend() before running your tests.");
    }
    return shared_backend;
}

void set_mock_hipdnn_backend(std::weak_ptr<Mock_hipdnn_backend> backend)
{
    auto shared_backend = backend.lock();
    if(!shared_backend)
    {
        throw std::invalid_argument("Cannot set a null mock hipDNN backend.");
    }

    mock_hipdnn_backend = std::move(backend);
}
}

hipdnnStatus_t hipdnnCreate(hipdnnHandle_t* handle)
{
    return fake_hipdnn_backend::backend()->hipdnnCreate(handle);
}

hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle)
{
    return fake_hipdnn_backend::backend()->hipdnnDestroy(handle);
}

hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipStream_t streamId)
{
    return fake_hipdnn_backend::backend()->hipdnnSetStream(handle, streamId);
}

hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle, hipStream_t* streamId)
{
    return fake_hipdnn_backend::backend()->hipdnnGetStream(handle, streamId);
}

hipdnnStatus_t hipdnnBackendCreateDescriptor(hipdnnBackendDescriptorType_t descriptor_type,
                                             hipdnnBackendDescriptor_t* descriptor)
{
    return fake_hipdnn_backend::backend()->hipdnnBackendCreateDescriptor(descriptor_type,
                                                                         descriptor);
}

hipdnnStatus_t hipdnnBackendDestroyDescriptor(hipdnnBackendDescriptor_t descriptor)
{
    return fake_hipdnn_backend::backend()->hipdnnBackendDestroyDescriptor(descriptor);
}

hipdnnStatus_t hipdnnBackendExecute(hipdnnHandle_t handle,
                                    hipdnnBackendDescriptor_t execution_plan,
                                    hipdnnBackendDescriptor_t variant_pack)
{
    return fake_hipdnn_backend::backend()->hipdnnBackendExecute(
        handle, execution_plan, variant_pack);
}

hipdnnStatus_t hipdnnBackendFinalize(hipdnnBackendDescriptor_t descriptor)
{
    return fake_hipdnn_backend::backend()->hipdnnBackendFinalize(descriptor);
}

hipdnnStatus_t hipdnnBackendGetAttribute(hipdnnBackendDescriptor_t descriptor,
                                         hipdnnBackendAttributeName_t attribute_name,
                                         hipdnnBackendAttributeType_t attribute_type,
                                         int64_t requested_element_count,
                                         int64_t* element_count,
                                         void* array_of_elements)
{
    return fake_hipdnn_backend::backend()->hipdnnBackendGetAttribute(descriptor,
                                                                     attribute_name,
                                                                     attribute_type,
                                                                     requested_element_count,
                                                                     element_count,
                                                                     array_of_elements);
}

hipdnnStatus_t hipdnnBackendSetAttribute(hipdnnBackendDescriptor_t descriptor,
                                         hipdnnBackendAttributeName_t attribute_name,
                                         hipdnnBackendAttributeType_t attribute_type,
                                         int64_t element_count,
                                         const void* array_of_elements)
{
    return fake_hipdnn_backend::backend()->hipdnnBackendSetAttribute(
        descriptor, attribute_name, attribute_type, element_count, array_of_elements);
}

const char* hipdnnGetErrorString(hipdnnStatus_t status)
{
    return fake_hipdnn_backend::backend()->hipdnnGetErrorString(status);
}

void hipdnnGetLastErrorString(char* message, size_t max_size)
{
    fake_hipdnn_backend::backend()->hipdnnGetLastErrorString(message, max_size);
}

hipdnnStatus_t hipdnnBackendCreateAndDeserializeGraph_ext(hipdnnBackendDescriptor_t* descriptor,
                                                          const uint8_t* serialized_graph,
                                                          size_t graph_byte_size)
{
    return fake_hipdnn_backend::backend()->hipdnnBackendCreateAndDeserializeGraph_ext(
        descriptor, serialized_graph, graph_byte_size);
}

//NOLINTEND
