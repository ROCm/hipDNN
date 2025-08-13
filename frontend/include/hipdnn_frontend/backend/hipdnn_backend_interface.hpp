// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <mutex>

#include <hipdnn_backend.h>

namespace hipdnn_frontend
{

class Hipdnn_backend_interface
{
public:
    virtual ~Hipdnn_backend_interface() = default;

    virtual hipdnnStatus_t create(hipdnnHandle_t* handle) = 0;
    virtual hipdnnStatus_t destroy(hipdnnHandle_t handle) = 0;
    virtual hipdnnStatus_t set_stream(hipdnnHandle_t handle, hipStream_t streamId) = 0;
    virtual hipdnnStatus_t get_stream(hipdnnHandle_t handle, hipStream_t* streamId) = 0;
    virtual hipdnnStatus_t backend_create_descriptor(hipdnnBackendDescriptorType_t descriptor_type,
                                                     hipdnnBackendDescriptor_t* descriptor)
        = 0;
    virtual hipdnnStatus_t backend_destroy_descriptor(hipdnnBackendDescriptor_t descriptor) = 0;
    virtual hipdnnStatus_t backend_execute(hipdnnHandle_t handle,
                                           hipdnnBackendDescriptor_t execution_plan,
                                           hipdnnBackendDescriptor_t variant_pack)
        = 0;
    virtual hipdnnStatus_t backend_finalize(hipdnnBackendDescriptor_t descriptor) = 0;
    virtual hipdnnStatus_t backend_get_attribute(hipdnnBackendDescriptor_t descriptor,
                                                 hipdnnBackendAttributeName_t attribute_name,
                                                 hipdnnBackendAttributeType_t attribute_type,
                                                 int64_t requested_element_count,
                                                 int64_t* element_count,
                                                 void* array_of_elements)
        = 0;
    virtual hipdnnStatus_t backend_set_attribute(hipdnnBackendDescriptor_t descriptor,
                                                 hipdnnBackendAttributeName_t attribute_name,
                                                 hipdnnBackendAttributeType_t attribute_type,
                                                 int64_t element_count,
                                                 const void* array_of_elements)
        = 0;
    virtual const char* get_error_string(hipdnnStatus_t status) = 0;
    virtual void get_last_error_string(char* message, size_t max_size) = 0;
    virtual hipdnnStatus_t
        backend_create_and_deserialize_graph_ext(hipdnnBackendDescriptor_t* descriptor,
                                                 const uint8_t* serialized_graph,
                                                 size_t graph_byte_size)
        = 0;
    virtual void logging_callback_ext(hipdnnSeverity_t severity, const char* msg) = 0;

    virtual hipdnnStatus_t set_engine_plugin_paths_ext(size_t num_paths,
                                                       const char* const* plugin_paths,
                                                       hipdnnPluginLoadingMode_ext_t mode)
        = 0;

    static inline std::shared_ptr<Hipdnn_backend_interface> backend_instance;
    static std::shared_ptr<Hipdnn_backend_interface> get_instance()
    {
        return backend_instance;
    }

    static void set_instance(std::shared_ptr<Hipdnn_backend_interface> instance)
    {
        backend_instance = std::move(instance);
    }

    static void reset_instance()
    {
        backend_instance.reset();
    }
};

}
