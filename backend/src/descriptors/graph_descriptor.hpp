// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include "backend_descriptor.hpp"
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_backend
{

class Graph_descriptor : public Backend_descriptor
{
private:
    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> _graph;

public:
    Graph_descriptor();
    ~Graph_descriptor() override = default;

    hipdnnStatus_t execute([[maybe_unused]] hipdnnHandle_t            handle,
                           [[maybe_unused]] hipdnnBackendDescriptor_t variant_pack) override;
    hipdnnStatus_t finalize() override;

    hipdnnStatus_t get_attribute([[maybe_unused]] hipdnnBackendAttributeName_t attribute_name,
                                 [[maybe_unused]] hipdnnBackendAttributeType_t attribute_type,
                                 [[maybe_unused]] int64_t  requested_element_count,
                                 [[maybe_unused]] int64_t* element_count,
                                 [[maybe_unused]] void*    array_of_elements) override;

    hipdnnStatus_t set_attribute([[maybe_unused]] hipdnnBackendAttributeName_t attribute_name,
                                 [[maybe_unused]] hipdnnBackendAttributeType_t attribute_type,
                                 [[maybe_unused]] int64_t                      element_count,
                                 [[maybe_unused]] const void* array_of_elements) override;

    hipdnnStatus_t deserialize_graph(const uint8_t* serialized_graph, size_t graph_byte_size);
};
}
