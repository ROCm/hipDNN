// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include "backend_descriptor.hpp"
#include "graph_generated.h"

namespace hipdnn_backend
{

class Graph_descriptor : public Backend_descriptor
{
private:
    std::unique_ptr<hipdnn::sdk::GraphT> _graph;

public:
    Graph_descriptor();
    ~Graph_descriptor() override = default;

    hipdnnStatus_t execute(hipdnnHandle_t handle, hipdnnBackendDescriptor_t variant_pack) override;
    hipdnnStatus_t finalize() override;

    hipdnnStatus_t get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                 hipdnnBackendAttributeType_t attribute_type,
                                 int64_t                      requested_element_count,
                                 int64_t*                     element_count,
                                 void*                        array_of_elements) override;

    hipdnnStatus_t set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                 hipdnnBackendAttributeType_t attribute_type,
                                 int64_t                      element_count,
                                 const void*                  array_of_elements) override;

    hipdnnStatus_t deserialize_graph(const uint8_t* serialized_graph, size_t graph_byte_size);
};
}
