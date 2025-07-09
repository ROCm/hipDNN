// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>
#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>
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

inline void unpack_serialized_engine_details(
    const void* buffer,
    size_t size,
    std::unique_ptr<hipdnn_sdk::data_objects::EngineDetailsT>& engine_details_out)
{
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
    if(!verifier.VerifyBuffer<hipdnn_sdk::data_objects::EngineDetails>())
    {
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                      "Invalid buffer: unable to verify the flatbuffer schema.");
    }

    auto engine_details = hipdnn_sdk::data_objects::UnPackEngineDetails(buffer);
    if(engine_details == nullptr)
    {
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_INTERNAL_ERROR,
                                      "Invalid buffer: unable to unpack the flatbuffer schema.");
    }

    engine_details_out = std::move(engine_details);
}

inline void unpack_serialized_engine_config(
    const void* buffer,
    size_t size,
    std::unique_ptr<hipdnn_sdk::data_objects::EngineConfigT>& engine_config_out)
{
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
    if(!verifier.VerifyBuffer<hipdnn_sdk::data_objects::EngineConfig>())
    {
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                      "Invalid buffer: unable to verify the flatbuffer schema.");
    }

    auto engine_config = hipdnn_sdk::data_objects::UnPackEngineConfig(buffer);
    if(engine_config == nullptr)
    {
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_INTERNAL_ERROR,
                                      "Invalid buffer: unable to unpack the flatbuffer schema.");
    }

    engine_config_out = std::move(engine_config);
}

} // namespace flatbuffer_utilities
} // namespace hipdnn_plugin