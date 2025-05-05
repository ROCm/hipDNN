// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "descriptors/graph_descriptor.hpp"
#include "descriptors/variant_descriptor.hpp"
#include "handle/handle.hpp"

namespace hipdnn_backend
{
struct Hipdnn_plugin_base
{
    virtual ~Hipdnn_plugin_base() = default;
    // Interface for plugin interaction using internal types.
    // Simplifies usage and integration with higher-level API handlers.
    virtual void execute(Graph_descriptor* graphdesc, Variant_descriptor* vpack, Handle* handle)
        = 0;
    virtual std::set<int64_t> get_applicable_engines(Graph_descriptor* graphdesc) = 0;
    virtual int64_t get_max_workspace_size(Graph_descriptor* graphdesc, int64_t engine_id) = 0;

    // Placeholder for additional plugin functionality.
    // Add pure virtual methods here as needed for plugin-specific operations.
    // These methods will be integrated into the CAPI in the future.
};
}