// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>

#include <memory>

namespace hipdnn_plugin
{
namespace flatbuffer_utilities
{

// Converts a serialized flatbuffer buffer into a hipdnn_plugin::PluginGraphT object.
// Throws on verification or unpacking failure.
inline void convert_serialized_plugin_graph_to_graph(
    const void* buffer, size_t size, std::unique_ptr<hipdnn_sdk::data_objects::GraphT>& graph_out)
{
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
    if(!verifier.VerifyBuffer<hipdnn_sdk::data_objects::Graph>())
    {
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                      "Invalid buffer: unable to verify the flatbuffer schema.");
    }

    auto graph = hipdnn_sdk::data_objects::UnPackGraph(buffer);
    if(graph == nullptr)
    {
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                      "Invalid buffer: unable to unpack the flatbuffer schema.");
    }

    graph_out = std::move(graph);
}

} // namespace flatbuffer_utilities
} // namespace hipdnn_plugin