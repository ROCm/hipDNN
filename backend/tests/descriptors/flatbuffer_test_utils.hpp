// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <string>
#include <vector>

namespace flatbuffer_test_utils
{

inline flatbuffers::FlatBufferBuilder create_valid_graph()
{
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
        tensor_attributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    flatbuffers::FlatBufferBuilder builder;
    auto graph_offset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      hipdnn_sdk::data_objects::DataType_HALF,
                                                      hipdnn_sdk::data_objects::DataType_BFLOAT16,
                                                      &tensor_attributes,
                                                      &nodes);
    builder.Finish(graph_offset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder create_valid_engine_details(int64_t engine_id)
{
    flatbuffers::FlatBufferBuilder builder;
    auto engine_details_offset = hipdnn_sdk::data_objects::CreateEngineDetails(builder, engine_id);
    builder.Finish(engine_details_offset);
    return builder;
}

}
