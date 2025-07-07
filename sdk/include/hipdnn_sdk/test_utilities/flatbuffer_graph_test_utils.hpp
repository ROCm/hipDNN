// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace flatbuffer_test_utils
{

using namespace hipdnn_sdk::data_objects;

inline flatbuffers::FlatBufferBuilder create_valid_graph()
{
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
        tensor_attributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    flatbuffers::FlatBufferBuilder builder;
    auto graph_offset = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                                    "test",
                                                                    DataType_FLOAT,
                                                                    DataType_HALF,
                                                                    DataType_BFLOAT16,
                                                                    &tensor_attributes,
                                                                    &nodes);
    builder.Finish(graph_offset);
    return builder;
}

inline hipdnnPluginConstData_t
    create_valid_const_data_graph(flatbuffers::DetachedBuffer& serialized_graph)
{
    hipdnnPluginConstData_t op_graph;
    op_graph.ptr = serialized_graph.data();
    op_graph.size = serialized_graph.size();
    return op_graph;
}

inline flatbuffers::FlatBufferBuilder create_valid_engine_details(int64_t engine_id)
{
    flatbuffers::FlatBufferBuilder builder;
    auto engine_details_offset = hipdnn_sdk::data_objects::CreateEngineDetails(builder, engine_id);
    builder.Finish(engine_details_offset);
    return builder;
}

inline hipdnnPluginConstData_t
    create_valid_const_data_engine_details(flatbuffers::DetachedBuffer& serialized_engine_details)
{
    hipdnnPluginConstData_t engine_details;
    engine_details.ptr = serialized_engine_details.data();
    engine_details.size = serialized_engine_details.size();
    return engine_details;
}

}