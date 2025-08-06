// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/backend/hipdnn_backend_interface.hpp>

namespace hipdnn_frontend
{

class Hipdnn_backend_wrapper : public Hipdnn_backend_interface
{
public:
    hipdnnStatus_t create(hipdnnHandle_t* handle) override
    {
        return hipdnnCreate(handle);
    }

    hipdnnStatus_t destroy(hipdnnHandle_t handle) override
    {
        return hipdnnDestroy(handle);
    }

    hipdnnStatus_t set_stream(hipdnnHandle_t handle, hipStream_t streamId) override
    {
        return hipdnnSetStream(handle, streamId);
    }

    hipdnnStatus_t get_stream(hipdnnHandle_t handle, hipStream_t* streamId) override
    {
        return hipdnnGetStream(handle, streamId);
    }

    hipdnnStatus_t backend_create_descriptor(hipdnnBackendDescriptorType_t descriptor_type,
                                             hipdnnBackendDescriptor_t* descriptor) override
    {
        return hipdnnBackendCreateDescriptor(descriptor_type, descriptor);
    }

    hipdnnStatus_t backend_destroy_descriptor(hipdnnBackendDescriptor_t descriptor) override
    {
        return hipdnnBackendDestroyDescriptor(descriptor);
    }

    hipdnnStatus_t backend_execute(hipdnnHandle_t handle,
                                   hipdnnBackendDescriptor_t execution_plan,
                                   hipdnnBackendDescriptor_t variant_pack) override
    {
        return hipdnnBackendExecute(handle, execution_plan, variant_pack);
    }

    hipdnnStatus_t backend_finalize(hipdnnBackendDescriptor_t descriptor) override
    {
        return hipdnnBackendFinalize(descriptor);
    }

    hipdnnStatus_t backend_get_attribute(hipdnnBackendDescriptor_t descriptor,
                                         hipdnnBackendAttributeName_t attribute_name,
                                         hipdnnBackendAttributeType_t attribute_type,
                                         int64_t requested_element_count,
                                         int64_t* element_count,
                                         void* array_of_elements) override
    {
        return hipdnnBackendGetAttribute(descriptor,
                                         attribute_name,
                                         attribute_type,
                                         requested_element_count,
                                         element_count,
                                         array_of_elements);
    }

    hipdnnStatus_t backend_set_attribute(hipdnnBackendDescriptor_t descriptor,
                                         hipdnnBackendAttributeName_t attribute_name,
                                         hipdnnBackendAttributeType_t attribute_type,
                                         int64_t element_count,
                                         const void* array_of_elements) override
    {
        return hipdnnBackendSetAttribute(
            descriptor, attribute_name, attribute_type, element_count, array_of_elements);
    }

    const char* get_error_string(hipdnnStatus_t status) override
    {
        return hipdnnGetErrorString(status);
    }

    void get_last_error_string(char* message, size_t max_size) override
    {
        hipdnnGetLastErrorString(message, max_size);
    }

    hipdnnStatus_t backend_create_and_deserialize_graph_ext(hipdnnBackendDescriptor_t* descriptor,
                                                            const uint8_t* serialized_graph,
                                                            size_t graph_byte_size) override
    {
        return hipdnnBackendCreateAndDeserializeGraph_ext(
            descriptor, serialized_graph, graph_byte_size);
    }

    void logging_callback_ext(hipdnnSeverity_t severity, const char* msg) override
    {
        hipdnnLoggingCallback_ext(severity, msg);
    }

    hipdnnStatus_t set_engine_plugin_paths_ext(size_t num_paths,
                                               const char* const* plugin_paths,
                                               hipdnnPluginLoadingMode_ext_t mode) override
    {
        return hipdnnSetEnginePluginPaths_ext(num_paths, plugin_paths, mode);
    }
};

static std::unique_ptr<Hipdnn_backend_interface> backend_wrapper;
static std::mutex backend_wrapper_mutex;

static Hipdnn_backend_interface& hipdnn_backend()
{
    if(!backend_wrapper)
    {
        std::lock_guard<std::mutex> lock(backend_wrapper_mutex);
        if(!backend_wrapper)
        {
            backend_wrapper = std::make_unique<Hipdnn_backend_wrapper>();
        }
    }
    return *backend_wrapper;
}

}
