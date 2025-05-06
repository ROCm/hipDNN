// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "flatbuffer_utilities.hpp"
#include "error.hpp"
#include "hipdnn_exception.hpp"
#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

namespace hipdnn_backend
{
namespace flatbuffer_utilities
{

void convert_serialized_graph_to_graph(const uint8_t* buffer,
                                       size_t size,
                                       std::unique_ptr<hipdnn_sdk::data_objects::GraphT>& graph_out)
{
    flatbuffers::Verifier verifier(buffer, size);
    if(!verifier.VerifyBuffer<hipdnn_sdk::data_objects::Graph>())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Invalid buffer: unable to verify the flatbuffer schema.");
    }

    auto graph = hipdnn_sdk::data_objects::UnPackGraph(buffer);
    if(graph == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_INTERNAL_ERROR,
                               "Invalid buffer: unable to unpack the flatbuffer schema.");
    }

    graph_out = std::move(graph);
}

} // namespace flatbuffer_utilities
} // namespace hipdnn_backend