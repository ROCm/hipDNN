// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "graph_generated.h"
#include "hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp"
#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/Json.hpp>

TEST(TestJson, SerializeGraphAsJson)
{
    auto graphBuilder = hipdnn_backend::test_utilities::createValidBatchnormGraph();
    auto graphFlatbuffer = graphBuilder.Release();
    auto graph = hipdnn_sdk::data_objects::GetGraph(graphFlatbuffer.data());

    auto graphJson = hipdnn_sdk::json::json(*graph);
    std::string out = graphJson.dump();
    std::cout << out << "\n";
}
