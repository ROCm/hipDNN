// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "graph_generated.h"
#include "hipdnn_status_t.h"
#include <memory>

namespace hipdnn_backend
{
namespace flatbuffer_utilities
{
hipdnnStatus_t convert_serialized_graph_to_graph(const uint8_t*                        buffer,
                                                 size_t                                size,
                                                 std::unique_ptr<hipdnn::sdk::GraphT>& graph_out);
}
}