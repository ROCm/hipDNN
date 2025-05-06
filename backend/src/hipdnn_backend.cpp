// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include "descriptors/backend_descriptor.hpp"
#include "descriptors/descriptor_factory.hpp"
#include "handle/handle.hpp"
#include "handle/handle_factory.hpp"
#include "helpers.hpp"
#include "hipdnn_exception.hpp"

#include <iostream>

using namespace hipdnn_backend;

namespace
{
void throw_if_invalid_descriptor(hipdnnBackendDescriptor_t descriptor)
{
    if(descriptor == nullptr)
    {
        throw hipdnn_backend::Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                                               "hipdnnBackendDescriptor_t is nullptr");
    }

    if(descriptor->type == HIPDNN_INVALID_TYPE)
    {
        throw hipdnn_backend::Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                                               "hipdnnBackendDescriptor_t is invalid type");
    }
}

template <typename T>
void throw_if_null(T* value)
{
    if(value == nullptr)
    {
        throw hipdnn_backend::Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                                               std::string(typeid(T).name()) + " is nullptr");
    }
}
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnCreate(hipdnnHandle_t* handle)
{
    return hipdnn_backend::try_catch([&] {
        throw_if_null(handle);
        hipdnn_backend::Handle_factory::create_handle(handle);
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle)
{
    return hipdnn_backend::try_catch([&] {
        throw_if_null(handle);

        delete handle;
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipStream_t streamId)
{
    return hipdnn_backend::try_catch([&] {
        throw_if_null(handle);

        handle->set_stream(streamId);
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle, hipStream_t* streamId)
{
    return hipdnn_backend::try_catch([&] {
        throw_if_null(handle);
        throw_if_null(streamId);

        *streamId = handle->get_stream();
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateDescriptor(
    hipdnnBackendDescriptorType_t descriptor_type, hipdnnBackendDescriptor_t* descriptor)
{
    return hipdnn_backend::try_catch(
        [&] { hipdnn_backend::Descriptor_factory::create(descriptor_type, descriptor); });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendDestroyDescriptor(hipdnnBackendDescriptor_t descriptor)
{
    return hipdnn_backend::try_catch([&] {
        throw_if_invalid_descriptor(descriptor);

        delete descriptor;
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendExecute([[maybe_unused]] hipdnnHandle_t handle,
                                                          hipdnnBackendDescriptor_t execution_plan,
                                                          hipdnnBackendDescriptor_t variant_pack)
{
    return hipdnn_backend::try_catch_with_status([&] {
        throw_if_invalid_descriptor(execution_plan);
        throw_if_invalid_descriptor(variant_pack);

        // TODO - add execute implementation
        return HIPDNN_STATUS_NOT_SUPPORTED;
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendFinalize(hipdnnBackendDescriptor_t descriptor)
{
    return hipdnn_backend::try_catch([&] {
        throw_if_invalid_descriptor(descriptor);

        descriptor->finalize();
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendGetAttribute(hipdnnBackendDescriptor_t descriptor,
                              hipdnnBackendAttributeName_t attribute_name,
                              hipdnnBackendAttributeType_t attribute_type,
                              int64_t requested_element_count,
                              int64_t* element_count,
                              void* array_of_elements)
{

    return hipdnn_backend::try_catch_with_status([&] {
        throw_if_invalid_descriptor(descriptor);

        return descriptor->get_attribute(attribute_name,
                                         attribute_type,
                                         requested_element_count,
                                         element_count,
                                         array_of_elements);
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendSetAttribute(hipdnnBackendDescriptor_t descriptor,
                              hipdnnBackendAttributeName_t attribute_name,
                              hipdnnBackendAttributeType_t attribute_type,
                              int64_t element_count,
                              const void* array_of_elements)
{
    return hipdnn_backend::try_catch_with_status([&] {
        throw_if_invalid_descriptor(descriptor);

        return descriptor->set_attribute(
            attribute_name, attribute_type, element_count, array_of_elements);
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateAndDeserializeGraph_ext(
    hipdnnBackendDescriptor_t* descriptor, const uint8_t* serialized_graph, size_t graph_byte_size)
{
    return hipdnn_backend::try_catch([&] {
        hipdnn_backend::Descriptor_factory::create_graph_ext(
            descriptor, serialized_graph, graph_byte_size);
    });
}
