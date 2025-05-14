// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include "descriptors/backend_descriptor.hpp"
#include "descriptors/descriptor_factory.hpp"
#include "error.hpp"
#include "handle/handle.hpp"
#include "handle/handle_factory.hpp"
#include "helpers.hpp"
#include "hipdnn_exception.hpp"
#include "plugin/plugin_manager.hpp"

#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/utilities/string_util.hpp>

using namespace hipdnn_backend;

#define LOG_API_ENTRY(format, ...) \
    HIPDNN_LOG_INFO("API called: [{}] " format, __func__ __VA_OPT__(, ) __VA_ARGS__)

#define LOG_API_SUCCESS(func_name, format, ...) \
    HIPDNN_LOG_INFO("API success: [{}] " format, func_name __VA_OPT__(, ) __VA_ARGS__)

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
} // namespace

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnCreate(hipdnnHandle_t* handle)
{
    LOG_API_ENTRY("handle_ptr={:p}", static_cast<void*>(handle));

    return hipdnn_backend::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);

        hipdnn_backend::Handle_factory::create_handle(handle);

        LOG_API_SUCCESS(api_name, "created_handle={:p}", static_cast<void*>(*handle));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle)
{
    LOG_API_ENTRY("handle={:p}", static_cast<void*>(handle));

    return hipdnn_backend::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);

        delete handle;

        LOG_API_SUCCESS(api_name, "");
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipStream_t stream_id)
{
    LOG_API_ENTRY(
        "handle={:p}, stream_id={:p}", static_cast<void*>(handle), static_cast<void*>(stream_id));

    return hipdnn_backend::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);

        handle->set_stream(stream_id);

        LOG_API_SUCCESS(api_name, "");
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle, hipStream_t* stream_id)
{
    LOG_API_ENTRY("handle={:p}, stream_id_ptr={:p}",
                  static_cast<void*>(handle),
                  static_cast<void*>(stream_id));

    return hipdnn_backend::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);
        throw_if_null(stream_id);

        *stream_id = handle->get_stream();

        LOG_API_SUCCESS(api_name, "retrieved_stream={:p}", static_cast<void*>(*stream_id));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateDescriptor(
    hipdnnBackendDescriptorType_t descriptor_type, hipdnnBackendDescriptor_t* descriptor)
{
    LOG_API_ENTRY("descriptor_type={}, descriptor_ptr={:p}",
                  hipdnn_backend::hipdnn_get_backend_descriptor_type_name(descriptor_type),
                  static_cast<void*>(descriptor));

    return hipdnn_backend::try_catch([&, api_name = __func__]() {
        hipdnn_backend::Descriptor_factory::create(descriptor_type, descriptor);

        LOG_API_SUCCESS(api_name, "created_descriptor={:p}", static_cast<void*>(*descriptor));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendDestroyDescriptor(hipdnnBackendDescriptor_t descriptor)
{
    LOG_API_ENTRY("descriptor={:p}", static_cast<void*>(descriptor));

    return hipdnn_backend::try_catch([&, api_name = __func__]() {
        throw_if_invalid_descriptor(descriptor);

        delete descriptor;

        LOG_API_SUCCESS(api_name, "");
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendExecute([[maybe_unused]] hipdnnHandle_t handle,
                                                          hipdnnBackendDescriptor_t execution_plan,
                                                          hipdnnBackendDescriptor_t variant_pack)
{
    LOG_API_ENTRY("handle={:p}, execution_plan={:p}, variant_pack={:p}",
                  static_cast<void*>(handle),
                  static_cast<void*>(execution_plan),
                  static_cast<void*>(variant_pack));

    return hipdnn_backend::try_catch([&]() {
        throw_if_invalid_descriptor(execution_plan);
        throw_if_invalid_descriptor(variant_pack);

        // TODO - add execute implementation
        throw Hipdnn_exception(HIPDNN_STATUS_NOT_SUPPORTED, "hipdnnBackendExecute not implemented");
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendFinalize(hipdnnBackendDescriptor_t descriptor)
{
    LOG_API_ENTRY("descriptor={:p}", static_cast<void*>(descriptor));

    return hipdnn_backend::try_catch([&, api_name = __func__]() {
        throw_if_invalid_descriptor(descriptor);

        descriptor->finalize();

        if(descriptor->type == HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR)
        {
            Plugin_manager plugin_manager;
            plugin_manager.initialize();
            plugin_manager.finalize_engine_config(descriptor);
        }

        LOG_API_SUCCESS(api_name, "");
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
    LOG_API_ENTRY("descriptor={:p}, attribute_name={}, attribute_type={}, "
                  "requested_element_count={}, element_count_ptr={:p}, array_of_elements_ptr={:p}",
                  static_cast<void*>(descriptor),
                  hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name),
                  hipdnn_backend::hipdnn_get_attribute_type_string(attribute_type),
                  requested_element_count,
                  static_cast<void*>(element_count),
                  array_of_elements);

    return hipdnn_backend::try_catch([&, api_name = __func__]() {
        throw_if_invalid_descriptor(descriptor);

        descriptor->get_attribute(attribute_name,
                                  attribute_type,
                                  requested_element_count,
                                  element_count,
                                  array_of_elements);

        LOG_API_SUCCESS(api_name,
                        "status={}, retrieved_element_count={}",
                        hipdnn_backend::hipdnn_get_status_string(HIPDNN_STATUS_SUCCESS),
                        *element_count);
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendSetAttribute(hipdnnBackendDescriptor_t descriptor,
                              hipdnnBackendAttributeName_t attribute_name,
                              hipdnnBackendAttributeType_t attribute_type,
                              int64_t element_count,
                              const void* array_of_elements)
{
    LOG_API_ENTRY("descriptor={:p}, attribute_name={}, attribute_type={}, "
                  "element_count={}, array_of_elements_ptr={:p}",
                  static_cast<void*>(descriptor),
                  hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name),
                  hipdnn_backend::hipdnn_get_attribute_type_string(attribute_type),
                  element_count,
                  array_of_elements);

    return hipdnn_backend::try_catch([&, api_name = __func__]() {
        throw_if_invalid_descriptor(descriptor);

        descriptor->set_attribute(attribute_name, attribute_type, element_count, array_of_elements);

        LOG_API_SUCCESS(
            api_name, "status={}", hipdnn_backend::hipdnn_get_status_string(HIPDNN_STATUS_SUCCESS));
    });
}

HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateAndDeserializeGraph_ext(
    hipdnnBackendDescriptor_t* descriptor, const uint8_t* serialized_graph, size_t graph_byte_size)
{
    LOG_API_ENTRY("descriptor_ptr={:p}, serialized_graph_ptr={:p}, graph_byte_size={}",
                  static_cast<void*>(descriptor),
                  static_cast<const void*>(serialized_graph),
                  graph_byte_size);

    return hipdnn_backend::try_catch([&, api_name = __func__]() {
        hipdnn_backend::Descriptor_factory::create_graph_ext(
            descriptor, serialized_graph, graph_byte_size);

        LOG_API_SUCCESS(api_name, "created_descriptor={:p}", static_cast<void*>(*descriptor));
    });
}

HIPDNN_BACKEND_EXPORT const char* hipdnnGetErrorString(hipdnnStatus_t status)
{
    LOG_API_ENTRY("status={}", hipdnn_backend::hipdnn_get_status_string(status));

    return hipdnn_backend::hipdnn_get_status_string(status);
}

HIPDNN_BACKEND_EXPORT void hipdnnGetLastErrorString(char* message, size_t max_size)
{
    LOG_API_ENTRY("message_ptr={:p}, max_size={}", static_cast<void*>(message), max_size);

    // Ignore status since API doesn't return it.
    // We still want to catch and log if the user provides incorrect parameters.
    auto _ = hipdnn_backend::try_catch([&, api_name = __func__] {
        throw_if_null(message);

        if(max_size == 0)
        {
            throw hipdnn_backend::Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM, "max_size is 0");
        }

        hipdnn::sdk::utilities::copy_max_size_with_null_terminator(
            message, hipdnn_backend::Last_error_manager::get_last_error(), max_size);

        LOG_API_SUCCESS(api_name, "set_error_message={:p}", static_cast<void*>(message));
    });
}
