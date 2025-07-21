// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace flatbuffer_test_utils
{

using namespace hipdnn_sdk::data_objects;

inline flatbuffers::FlatBufferBuilder create_empty_valid_graph()
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

inline flatbuffers::FlatBufferBuilder
    create_valid_batchnorm_graph(std::vector<int64_t> strides = {1, 3, 224, 224},
                                 std::vector<int64_t> dims = {1, 3, 224, 224},
                                 bool has_optional_attributes = true)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
        tensor_attributes;

    tensor_attributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));

    tensor_attributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));

    tensor_attributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "scale", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));

    tensor_attributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "bias", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));

    if(has_optional_attributes)
    {
        tensor_attributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 5, "est_mean", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));

        tensor_attributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 6, "est_variance", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));
    }

    auto bnorm_attributes = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder,
        1, // x uid
        has_optional_attributes ? flatbuffers::Optional<int64_t>(5)
                                : flatbuffers::nullopt, // mean uid
        has_optional_attributes ? flatbuffers::Optional<int64_t>(6)
                                : flatbuffers::nullopt, // inv_variance uid
        3, // scale uid
        4, // bias uid
        2 // y uid
    );

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm",
        hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes,
        bnorm_attributes.Union());
    nodes.push_back(node);

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

inline flatbuffers::FlatBufferBuilder create_valid_engine_config(int64_t config_id)
{
    flatbuffers::FlatBufferBuilder builder;
    auto engine_config_offset = hipdnn_sdk::data_objects::CreateEngineConfig(builder, config_id);
    builder.Finish(engine_config_offset);
    return builder;
}

inline hipdnnPluginConstData_t
    create_valid_const_data_engine_config(flatbuffers::DetachedBuffer& serialized_engine_config)
{
    hipdnnPluginConstData_t engine_config;
    engine_config.ptr = serialized_engine_config.data();
    engine_config.size = serialized_engine_config.size();
    return engine_config;
}

}