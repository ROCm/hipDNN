// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include "descriptors/backend_descriptor.hpp"
#include "descriptors/descriptor_factory.hpp"
#include "handle/handle.hpp"
#include "handle/handle_factory.hpp"
#include "helpers.hpp"

#include <iostream>

using namespace hipdnn_backend;

namespace
{
hipdnnStatus_t is_valid_descriptor(hipdnnBackendDescriptor_t descriptor)
{
    if(descriptor == nullptr)
    {
        return HIPDNN_STATUS_BAD_PARAM;
    }

    if(descriptor->type == HIPDNN_INVALID_TYPE)
    {
        return HIPDNN_STATUS_BAD_PARAM;
    }

    return HIPDNN_STATUS_SUCCESS;
}
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnCreate(hipdnnHandle_t* handle)
{
    return hipdnn_backend::try_catch(
        [&] { return hipdnn_backend::Handle_factory::create_handle(handle); }, false);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle)
{
    return hipdnn_backend::try_catch(
        [&] {
            if(handle == nullptr)
            {
                return HIPDNN_STATUS_BAD_PARAM;
            }
            delete handle;

            return HIPDNN_STATUS_SUCCESS;
        },
        false);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipStream_t streamId)
{
    return hipdnn_backend::try_catch(
        [&] {
            if(handle != nullptr)
            {
                auto backend_handle = dynamic_cast<hipdnn_backend::Handle*>(handle);
                if(backend_handle == nullptr)
                {
                    return HIPDNN_STATUS_BAD_PARAM;
                }

                backend_handle->set_stream(streamId);

                return HIPDNN_STATUS_SUCCESS;
            }
            return HIPDNN_STATUS_BAD_PARAM;
        },
        false);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle, hipStream_t* streamId)
{
    return hipdnn_backend::try_catch(
        [&] {
            if(handle == nullptr || streamId == nullptr)
            {
                return HIPDNN_STATUS_BAD_PARAM;
            }

            auto backend_handle = dynamic_cast<hipdnn_backend::Handle*>(handle);
            if(backend_handle == nullptr)
            {
                return HIPDNN_STATUS_BAD_PARAM;
            }

            *streamId = backend_handle->get_stream();
            return HIPDNN_STATUS_SUCCESS;
        },
        false);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateDescriptor(
    hipdnnBackendDescriptorType_t descriptor_type, hipdnnBackendDescriptor_t* descriptor)
{
    return hipdnn_backend::try_catch(
        [&] { return hipdnn_backend::Descriptor_factory::create(descriptor_type, descriptor); },
        false);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendDestroyDescriptor(hipdnnBackendDescriptor_t descriptor)
{
    return hipdnn_backend::try_catch(
        [&] {
            if(descriptor == nullptr)
            {
                return HIPDNN_STATUS_BAD_PARAM;
            }

            delete descriptor;

            return HIPDNN_STATUS_SUCCESS;
        },
        false);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendExecute([[maybe_unused]] hipdnnHandle_t handle,
                                                          hipdnnBackendDescriptor_t execution_plan,
                                                          hipdnnBackendDescriptor_t variant_pack)
{
    return hipdnn_backend::try_catch(
        [&] {
            hipdnnStatus_t status = is_valid_descriptor(execution_plan);
            if(status != HIPDNN_STATUS_SUCCESS)
            {
                return status;
            }

            status = is_valid_descriptor(variant_pack);
            if(status != HIPDNN_STATUS_SUCCESS)
            {
                return status;
            }

            return HIPDNN_STATUS_NOT_SUPPORTED;
        },
        false);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendFinalize(hipdnnBackendDescriptor_t descriptor)
{
    return hipdnn_backend::try_catch(
        [&] {
            hipdnnStatus_t status = is_valid_descriptor(descriptor);
            if(status != HIPDNN_STATUS_SUCCESS)
            {
                return status;
            }

            auto backend_descriptor = dynamic_cast<hipdnn_backend::Backend_descriptor*>(descriptor);
            if(backend_descriptor == nullptr)
            {
                return HIPDNN_STATUS_BAD_PARAM;
            }

            return backend_descriptor->finalize();
        },
        false);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendGetAttribute(hipdnnBackendDescriptor_t descriptor,
                              hipdnnBackendAttributeName_t attribute_name,
                              hipdnnBackendAttributeType_t attribute_type,
                              int64_t requested_element_count,
                              int64_t* element_count,
                              void* array_of_elements)
{

    return hipdnn_backend::try_catch(
        [&] {
            hipdnnStatus_t status = is_valid_descriptor(descriptor);
            if(status != HIPDNN_STATUS_SUCCESS)
            {
                return status;
            }

            auto backend_descriptor = dynamic_cast<hipdnn_backend::Backend_descriptor*>(descriptor);
            if(backend_descriptor == nullptr)
            {
                return HIPDNN_STATUS_BAD_PARAM;
            }

            return backend_descriptor->get_attribute(attribute_name,
                                                     attribute_type,
                                                     requested_element_count,
                                                     element_count,
                                                     array_of_elements);
        },
        false);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendSetAttribute(hipdnnBackendDescriptor_t descriptor,
                              hipdnnBackendAttributeName_t attribute_name,
                              hipdnnBackendAttributeType_t attribute_type,
                              int64_t element_count,
                              const void* array_of_elements)
{
    return hipdnn_backend::try_catch(
        [&] {
            hipdnnStatus_t status = is_valid_descriptor(descriptor);
            if(status != HIPDNN_STATUS_SUCCESS)
            {
                return status;
            }

            auto backend_descriptor = dynamic_cast<hipdnn_backend::Backend_descriptor*>(descriptor);
            if(backend_descriptor == nullptr)
            {
                return HIPDNN_STATUS_BAD_PARAM;
            }

            return backend_descriptor->set_attribute(
                attribute_name, attribute_type, element_count, array_of_elements);
        },
        false);
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateAndDeserializeGraph_ext(
    hipdnnBackendDescriptor_t* descriptor, const uint8_t* serialized_graph, size_t graph_byte_size)
{
    return hipdnn_backend::try_catch(
        [&] {
            return hipdnn_backend::Descriptor_factory::create_graph_ext(
                descriptor, serialized_graph, graph_byte_size);
        },
        false);
}
