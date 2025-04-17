// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "graph_generated.h"
#include <flatbuffers/flatbuffers.h>
#include <string>
#include <vector>

namespace flatbuffer_test_utils
{

static flatbuffers::FlatBufferBuilder create_valid_graph()
{
    std::vector<::flatbuffers::Offset<hipdnn::sdk::TensorAttributes>> tensor_attributes;
    std::vector<::flatbuffers::Offset<hipdnn::sdk::Node>>             nodes;
    flatbuffers::FlatBufferBuilder                                    builder;
    auto graph_offset = hipdnn::sdk::CreateGraphDirect(builder,
                                                       "test",
                                                       hipdnn::sdk::DataType_FLOAT,
                                                       hipdnn::sdk::DataType_HALF,
                                                       hipdnn::sdk::DataType_BFLOAT16,
                                                       &tensor_attributes,
                                                       &nodes);
    builder.Finish(graph_offset);
    return builder;
}

}