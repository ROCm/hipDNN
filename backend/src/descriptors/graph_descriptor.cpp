// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "graph_descriptor.hpp"
#include "error.hpp"
#include "flatbuffer_utilities.hpp"
#include "hipdnn_backend_descriptor_type.h"
#include "hipdnn_exception.hpp"
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_backend
{

void Graph_descriptor::finalize()
{
    THROW_IF_NULL(_graph, HIPDNN_STATUS_BAD_PARAM, "Graph_descriptor::finalize: graph is null");
    THROW_IF_NULL(_handle, HIPDNN_STATUS_BAD_PARAM, "Graph_descriptor::finalize: handle is null");
    hipdnnBackendDescriptorImpl<Graph_descriptor>::finalize();
}

void Graph_descriptor::set_handle(hipdnnBackendAttributeType_t attribute_type,
                                  int64_t element_count,
                                  const void* array_of_elements)
{
    THROW_IF_NE(attribute_type,
                HIPDNN_TYPE_HANDLE,
                HIPDNN_STATUS_BAD_PARAM,
                "Graph_descriptor failed to set handle: Invalid attribute type.");
    THROW_IF_NE(element_count,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "Graph_descriptor failed to set handle: Invalid element count.");
    THROW_IF_NULL(array_of_elements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Graph_descriptor failed to set handle: Null pointer.");

    hipdnnHandle_t handle = *static_cast<const hipdnnHandle_t*>(array_of_elements);

    THROW_IF_NULL(handle,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Graph_descriptor failed to set handle: Handle is null.");

    _handle = handle;
}

void Graph_descriptor::get_attribute([[maybe_unused]] hipdnnBackendAttributeName_t attribute_name,
                                     [[maybe_unused]] hipdnnBackendAttributeType_t attribute_type,
                                     [[maybe_unused]] int64_t requested_element_count,
                                     [[maybe_unused]] int64_t* element_count,
                                     [[maybe_unused]] void* array_of_elements) const
{
    throw Hipdnn_exception(HIPDNN_STATUS_NOT_SUPPORTED,
                           "Graph_descriptor::get_attribute: not supported");
}

void Graph_descriptor::set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                     hipdnnBackendAttributeType_t attribute_type,
                                     int64_t element_count,
                                     const void* array_of_elements)
{
    THROW_IF_TRUE(is_finalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "Graph_descriptor::set_attribute() failed: Already finalized.");

    switch(attribute_name)
    {
    case HIPDNN_ATTR_OPERATIONGRAPH_HANDLE:
        set_handle(attribute_type, element_count, array_of_elements);
        break;
    default:
        throw Hipdnn_exception(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("Graph_descriptor::set_attribute() is not supported for attribute ")
                + hipdnn_backend::hipdnn_get_attribute_name_string(attribute_name) + ".");
    }

    if(attribute_name != HIPDNN_ATTR_OPERATIONGRAPH_HANDLE)
    {
        // Clear the serialized graph when the graph is modified
        _serialized_graph.clear();
    }
}

void Graph_descriptor::deserialize_graph(const uint8_t* serialized_graph, size_t graph_byte_size)
{
    THROW_IF_NULL(serialized_graph,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Graph_descriptor::deserialize_graph: serialized_graph is null");
    THROW_IF_TRUE(graph_byte_size == 0,
                  HIPDNN_STATUS_BAD_PARAM,
                  "Graph_descriptor::deserialize_graph: graph_byte_size is 0");

    // TODO: Consider lazy graph deserialization
    flatbuffer_utilities::convert_serialized_graph_to_graph(
        serialized_graph, graph_byte_size, _graph);

    // Save the serialized graph for later use
    _serialized_graph.assign(serialized_graph, serialized_graph + graph_byte_size);
}

const std::vector<uint8_t>& Graph_descriptor::get_serialized_graph()
{
    if(_serialized_graph.empty())
    {
        THROW_IF_NULL(_graph,
                      HIPDNN_STATUS_INTERNAL_ERROR,
                      "Graph_descriptor::get_serialized_graph: graph is null");

        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(hipdnn_sdk::data_objects::Graph::Pack(builder, _graph.get()));

        const uint8_t* serialized_graph = builder.GetBufferPointer();
        size_t graph_byte_size = builder.GetSize();

        _serialized_graph.assign(serialized_graph, serialized_graph + graph_byte_size);
    }

    return _serialized_graph;
}

hipdnnBackendDescriptorType_t Graph_descriptor::get_static_type()
{
    return HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR;
}

} // namespace hipdnn_backend
