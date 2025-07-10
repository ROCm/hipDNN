/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include "engines/solvers/miopen_batchnorm_solver.hpp"

#include <gtest/gtest.h>
#include <numeric>

#include <hipdnn_sdk/data_objects/graph_generated.h>

//remove this later
#include "hipdnn_engine_plugin_handle.hpp"
#include "miopen_handle_factory.hpp"
#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>

using namespace miopen_legacy_plugin;

class Test_miopen_batchnorm_solver : public ::testing::Test
{
protected:
    Miopen_batchnorm_solver solver;
    hipdnnEnginePluginHandle dummy_handle;
};

TEST_F(Test_miopen_batchnorm_solver, IsApplicableReturnsTrue)
{
    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    auto graph_fb
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Graph>(builder.GetBufferPointer());

    EXPECT_TRUE(solver.is_applicable(*graph_fb));
}

TEST_F(Test_miopen_batchnorm_solver, GetWorkspaceSizeReturnsExpectedValue)
{
    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    auto graph_fb
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Graph>(builder.GetBufferPointer());

    size_t workspace_size = solver.get_workspace_size(dummy_handle, *graph_fb);

    EXPECT_EQ(workspace_size, 0u);
}
