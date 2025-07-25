// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <vector>

#include "backend_descriptor.hpp"
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_backend
{

class Graph_descriptor : public hipdnnBackendDescriptorImpl<Graph_descriptor>
{
private:
    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> _graph;
    hipdnnHandle_t _handle = nullptr;
    std::vector<uint8_t> _serialized_graph;

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

    const std::vector<uint8_t>& get_serialized_graph();

    static hipdnnBackendDescriptorType_t get_static_type();
};
}
