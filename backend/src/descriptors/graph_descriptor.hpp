// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "backend_descriptor.hpp"
#include <flatbuffers/detached_buffer.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>
#include <memory>
#include <vector>

namespace hipdnn_backend
{

class Graph_descriptor : public hipdnnBackendDescriptorImpl<Graph_descriptor>
{
private:
    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> _graph;
    hipdnnHandle_t _handle = nullptr;
    mutable flatbuffers::DetachedBuffer _graph_serialized_buffer;

    void set_handle(hipdnnBackendAttributeType_t attribute_type,
                    int64_t element_count,
                    const void* array_of_elements);

public:
    void finalize() override;

    void get_attribute([[maybe_unused]] hipdnnBackendAttributeName_t attribute_name,
                       [[maybe_unused]] hipdnnBackendAttributeType_t attribute_type,
                       [[maybe_unused]] int64_t requested_element_count,
                       [[maybe_unused]] int64_t* element_count,
                       [[maybe_unused]] void* array_of_elements) const override;

    void set_attribute([[maybe_unused]] hipdnnBackendAttributeName_t attribute_name,
                       [[maybe_unused]] hipdnnBackendAttributeType_t attribute_type,
                       [[maybe_unused]] int64_t element_count,
                       [[maybe_unused]] const void* array_of_elements) override;

    void deserialize_graph(const uint8_t* serialized_graph, size_t graph_byte_size);

    virtual hipdnnPluginConstData_t get_serialized_graph() const;
    virtual hipdnnHandle_t get_handle() const;

    static hipdnnBackendDescriptorType_t get_static_type();
};
}
