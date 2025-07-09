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

#define HIP_CHECK(status)                                                                      \
    do                                                                                         \
    {                                                                                          \
        if(status != hipSuccess)                                                               \
        {                                                                                      \
            std::cerr << "HIP Error: " << hipGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << "\n";                                      \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while(0)

using namespace miopen_legacy_plugin;

class Test_miopen_batchnorm_solver : public ::testing::Test
{
protected:
    Miopen_batchnorm_solver solver;
    hipdnn_sdk::data_objects::GraphT op_graph;
    hipdnnEnginePluginHandle dummy_handle;
};

TEST_F(Test_miopen_batchnorm_solver, IsApplicableReturnsTrue)
{
    EXPECT_TRUE(solver.is_applicable(op_graph));
}

TEST_F(Test_miopen_batchnorm_solver, GetWorkspaceSizeReturnsExpectedValue)
{
    size_t workspace_size = solver.get_workspace_size(dummy_handle, op_graph);
    EXPECT_EQ(workspace_size, 0u);
}

TEST_F(Test_miopen_batchnorm_solver, ExecuteWithValidInputs)
{
    flatbuffers::FlatBufferBuilder builder;

    // Create device buffers
    std::vector<hipdnnPluginDeviceBuffer_t> device_buffers;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
        tensor_attributes;

    std::vector<int64_t> strides = {1, 3, 224, 224}; // always in nchw
    std::vector<int64_t> dims = {1, 3, 224, 224};

    tensor_attributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));
    hipdnnPluginDeviceBuffer_t x_buffer;
    x_buffer.uid = 1;
    size_t x_buffer_size
        = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    HIP_CHECK(hipMalloc(&x_buffer.ptr, static_cast<size_t>(x_buffer_size * sizeof(float))));
    device_buffers.push_back(x_buffer);

    tensor_attributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));
    hipdnnPluginDeviceBuffer_t y_buffer;
    y_buffer.uid = 2;
    size_t y_buffer_size
        = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    HIP_CHECK(hipMalloc(&y_buffer.ptr, y_buffer_size * sizeof(float)));
    device_buffers.push_back(y_buffer);

    tensor_attributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "scale", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));
    hipdnnPluginDeviceBuffer_t scale_buffer;
    scale_buffer.uid = 3;
    size_t scale_buffer_size
        = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    HIP_CHECK(hipMalloc(&scale_buffer.ptr, scale_buffer_size * sizeof(float)));
    device_buffers.push_back(scale_buffer);

    tensor_attributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "bias", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));
    hipdnnPluginDeviceBuffer_t bias_buffer;
    bias_buffer.uid = 4;
    size_t bias_buffer_size
        = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    HIP_CHECK(hipMalloc(&bias_buffer.ptr, bias_buffer_size * sizeof(float)));
    device_buffers.push_back(bias_buffer);

    tensor_attributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 5, "est_mean", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));
    hipdnnPluginDeviceBuffer_t mean_buffer;
    mean_buffer.uid = 5;
    size_t mean_buffer_size
        = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    HIP_CHECK(hipMalloc(&mean_buffer.ptr, mean_buffer_size * sizeof(float)));
    device_buffers.push_back(mean_buffer);

    tensor_attributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 6, "est_variance", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));
    hipdnnPluginDeviceBuffer_t variance_buffer;
    variance_buffer.uid = 6;
    size_t variance_buffer_size
        = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    HIP_CHECK(hipMalloc(&variance_buffer.ptr, variance_buffer_size * sizeof(float)));
    device_buffers.push_back(variance_buffer);

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    auto bnorm_attributes
        = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(builder,
                                                                       1, // x uid
                                                                       5, // mean uid
                                                                       6, // inv_variance uid
                                                                       3, // scale uid
                                                                       4, // bias uid
                                                                       2 // y uid
        );

    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm",
        hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes,
        bnorm_attributes.Union());
    nodes.push_back(node);

    auto graph_offset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test_batch_norm_graph",
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      hipdnn_sdk::data_objects::DataType_HALF,
                                                      hipdnn_sdk::data_objects::DataType_BFLOAT16,
                                                      &tensor_attributes,
                                                      &nodes);

    builder.Finish(graph_offset);

    //alloc buffferrrs

    hipdnnEnginePluginHandle_t handle;
    miopen_legacy_plugin::Miopen_handle_factory::create_miopen_handle(&handle);
    //todo might need stream.

    // Get the buffer pointer and size from the builder
    const uint8_t* buffer_ptr = builder.GetBufferPointer();
    // Get a pointer to the root flatbuffer object
    auto graph_fb = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Graph>(buffer_ptr);
    // Unpack to native GraphT object
    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph(graph_fb->UnPack());

    solver.execute_graph(*handle,
                         *graph,
                         device_buffers.data(), // device_buffers
                         static_cast<uint32_t>(device_buffers.size()), // num_device_buffers
                         nullptr // workspace
    );

    for(auto& buffer : device_buffers)
    {
        HIP_CHECK(hipFree(buffer.ptr));
    }
}
