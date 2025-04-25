// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "graph_descriptor.hpp"
#include "error.hpp"
#include "flatbuffer_utilities.hpp"
#include "hipdnn_backend_descriptor_type.h"
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_backend
{

Graph_descriptor::Graph_descriptor()
{
    type = HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR;
}

hipdnnStatus_t Graph_descriptor::finalize()
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t
    Graph_descriptor::get_attribute([[maybe_unused]] hipdnnBackendAttributeName_t attribute_name,
                                    [[maybe_unused]] hipdnnBackendAttributeType_t attribute_type,
                                    [[maybe_unused]] int64_t  requested_element_count,
                                    [[maybe_unused]] int64_t* element_count,
                                    [[maybe_unused]] void*    array_of_elements)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t
    Graph_descriptor::set_attribute([[maybe_unused]] hipdnnBackendAttributeName_t attribute_name,
                                    [[maybe_unused]] hipdnnBackendAttributeType_t attribute_type,
                                    [[maybe_unused]] int64_t                      element_count,
                                    [[maybe_unused]] const void*                  array_of_elements)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t Graph_descriptor::deserialize_graph(const uint8_t* serialized_graph,
                                                   size_t         graph_byte_size)
{
    if(serialized_graph == nullptr || graph_byte_size == 0)
    {
        return set_last_error(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                              "Graph_descriptor::deserialize_graph: serialized_graph is null or "
                              "graph_byte_size is 0");
    }

    auto status = flatbuffer_utilities::convert_serialized_graph_to_graph(
        serialized_graph, graph_byte_size, _graph);

    if(status != HIPDNN_STATUS_SUCCESS || _graph == nullptr)
    {
        return set_last_error(status,
                              "Graph_descriptor::deserialize_graph: failed to convert serialized "
                              "graph to graph");
    }

    return HIPDNN_STATUS_SUCCESS;
}
}