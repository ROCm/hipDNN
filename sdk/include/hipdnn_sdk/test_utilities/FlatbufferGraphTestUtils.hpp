// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>

namespace hipdnn_backend::test_utilities
{

using namespace hipdnn_sdk::data_objects;

inline flatbuffers::FlatBufferBuilder createEmptyValidGraph()
{
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    flatbuffers::FlatBufferBuilder builder;
    auto graphOffset = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                                   "test",
                                                                   DataType_FLOAT,
                                                                   DataType_HALF,
                                                                   DataType_BFLOAT16,
                                                                   &tensorAttributes,
                                                                   &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createValidBatchnormGraph(std::vector<int64_t> strides = {1, 3, 224, 224},
                              std::vector<int64_t> dims = {1, 3, 224, 224},
                              bool hasOptionalAttributes = true,
                              hipdnn_sdk::data_objects::DataType inputDataType = DataType_FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    std::vector<int64_t> derivedStrides = {1, strides[1], 1, 1};
    std::vector<int64_t> derivedDims = {1, dims[1], 1, 1};

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_sdk::data_objects::DataType_FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_sdk::data_objects::DataType_FLOAT,
        &derivedStrides,
        &derivedDims));

    if(hasOptionalAttributes)
    {
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            5,
            "est_mean",
            hipdnn_sdk::data_objects::DataType_FLOAT,
            &derivedStrides,
            &derivedDims));

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            6,
            "est_variance",
            hipdnn_sdk::data_objects::DataType_FLOAT,
            &derivedStrides,
            &derivedDims));
    }

    auto bnormAttributes = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder,
        1, // x uid
        hasOptionalAttributes ? flatbuffers::Optional<int64_t>(5)
                              : flatbuffers::nullopt, // mean uid
        hasOptionalAttributes ? flatbuffers::Optional<int64_t>(6)
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
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                                   "test",
                                                                   DataType_FLOAT,
                                                                   DataType_HALF,
                                                                   DataType_BFLOAT16,
                                                                   &tensorAttributes,
                                                                   &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createValidBatchnormBwdGraph(std::vector<int64_t> strides = {1, 3, 224, 224},
                                 std::vector<int64_t> dims = {1, 3, 224, 224},
                                 bool hasOptionalAttributes = true,
                                 hipdnn_sdk::data_objects::DataType inputDataType = DataType_FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    std::vector<int64_t> derivedStrides = {1, strides[1], 1, 1};
    std::vector<int64_t> derivedDims = {1, dims[1], 1, 1};

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "dy", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "dx", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "scale",
        hipdnn_sdk::data_objects::DataType_FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "dscale",
        hipdnn_sdk::data_objects::DataType_FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "dbias",
        hipdnn_sdk::data_objects::DataType_FLOAT,
        &derivedStrides,
        &derivedDims));

    if(hasOptionalAttributes)
    {
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            7,
            "mean",
            hipdnn_sdk::data_objects::DataType_FLOAT,
            &derivedStrides,
            &derivedDims));

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            8,
            "inv_variance",
            hipdnn_sdk::data_objects::DataType_FLOAT,
            &derivedStrides,
            &derivedDims));
    }

    auto bnormAttributes = hipdnn_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        2, // dy_tensor_uid
        1, // x_tensor_uid
        hasOptionalAttributes ? flatbuffers::Optional<int64_t>(7)
                              : flatbuffers::nullopt, // mean_tensor_uid
        hasOptionalAttributes ? flatbuffers::Optional<int64_t>(8)
                              : flatbuffers::nullopt, // inv_variance_tensor_uid
        4, // scale_tensor_uid
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(), // peer_stats_tensor_uid
        3, // dx_tensor_uid
        5, // dscale_tensor_uid
        6 // dbias_tensor_uid
    );

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_bwd",
        hipdnn_sdk::data_objects::NodeAttributes_BatchnormBackwardAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                                   "test",
                                                                   DataType_FLOAT,
                                                                   DataType_HALF,
                                                                   DataType_BFLOAT16,
                                                                   &tensorAttributes,
                                                                   &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createValidConvFwdGraph(std::vector<int64_t> xDims = {1, 1, 1, 1},
                            std::vector<int64_t> xStrides = {1, 1, 1, 1},
                            std::vector<int64_t> wDims = {1, 1, 1, 1},
                            std::vector<int64_t> wStrides = {1, 1, 1, 1},
                            std::vector<int64_t> yDims = {1, 1, 1, 1},
                            std::vector<int64_t> yStrides = {1, 1, 1, 1},
                            std::vector<int64_t> convPrePadding = {0, 0},
                            std::vector<int64_t> convPostPadding = {0, 0},
                            std::vector<int64_t> convStrides = {1, 1},
                            std::vector<int64_t> convDilation = {1, 1},
                            hipdnn_sdk::data_objects::DataType dataType = DataType_FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", dataType, &xStrides, &xDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "w", dataType, &wStrides, &wDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "y", dataType, &yStrides, &yDims));

    auto convAttributes = hipdnn_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder,
        1, // x tensor uid
        2, // w tensor uid
        3, // y tensor uid
        &convPrePadding,
        &convPostPadding,
        &convStrides,
        &convDilation,
        hipdnn_sdk::data_objects::ConvMode_CROSS_CORRELATION
    );

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv_fwd",
        hipdnn_sdk::data_objects::NodeAttributes_ConvolutionFwdAttributes,
        convAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                                   "test",
                                                                   DataType_FLOAT,
                                                                   DataType_FLOAT,
                                                                   DataType_FLOAT,
                                                                   &tensorAttributes,
                                                                   &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline hipdnnPluginConstData_t
    createValidConstDataGraph(flatbuffers::DetachedBuffer& serializedGraph)
{
    hipdnnPluginConstData_t opGraph;
    opGraph.ptr = serializedGraph.data();
    opGraph.size = serializedGraph.size();
    return opGraph;
}

inline flatbuffers::FlatBufferBuilder createValidEngineDetails(int64_t engineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto engineDetailsOffset = hipdnn_sdk::data_objects::CreateEngineDetails(builder, engineId);
    builder.Finish(engineDetailsOffset);
    return builder;
}

inline hipdnnPluginConstData_t
    createValidConstDataEngineDetails(flatbuffers::DetachedBuffer& serializedEngineDetails)
{
    hipdnnPluginConstData_t engineDetails;
    engineDetails.ptr = serializedEngineDetails.data();
    engineDetails.size = serializedEngineDetails.size();
    return engineDetails;
}

inline flatbuffers::FlatBufferBuilder createValidEngineConfig(int64_t configId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto engineConfigOffset = hipdnn_sdk::data_objects::CreateEngineConfig(builder, configId);
    builder.Finish(engineConfigOffset);
    return builder;
}

inline hipdnnPluginConstData_t
    createValidConstDataEngineConfig(flatbuffers::DetachedBuffer& serializedEngineConfig)
{
    hipdnnPluginConstData_t engineConfig;
    engineConfig.ptr = serializedEngineConfig.data();
    engineConfig.size = serializedEngineConfig.size();
    return engineConfig;
}

}
