// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "graph_generated.h"
#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>

using namespace hipdnn::sdk;

TEST(FlatBuffersFrontendTests, SerializeAndDeserialize)
{
    flatbuffers::FlatBufferBuilder builder;

    auto graph = CreateGraphDirect(builder, "Graph");
    builder.Finish(graph);

    auto deserialized_graph = flatbuffers::GetRoot<Graph>(builder.GetBufferPointer());
    EXPECT_EQ(deserialized_graph->name()->str(), "Graph");
}

TEST(FrontendAPITests, CreatePointwiseAttributes)
{
    hipdnn_frontend::graph::Pointwise_attributes pointwise_attributes;

    pointwise_attributes.set_input_0(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    pointwise_attributes.set_output_0(
        std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    pointwise_attributes.set_operation(hipdnn_frontend::graph::PointwiseMode_t::RELU)
        .set_relu_lower_clip(0.1f)
        .set_relu_upper_clip(6.0f)
        .set_relu_lower_slope(0.01f)
        .set_axis(1);

    auto input_tensor = pointwise_attributes.get_input_0();
    EXPECT_FALSE(input_tensor->has_uid());

    input_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = pointwise_attributes.get_output_0();
    EXPECT_FALSE(output_tensor->has_uid());

    output_tensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::HALF)
        .set_dim({5, 6, 7, 8})
        .set_stride({1, 2, 3, 4});

    EXPECT_EQ(input_tensor->get_uid(), 1);
    EXPECT_EQ(input_tensor->get_name(), "InputTensor");
    EXPECT_EQ(input_tensor->get_data_type(), hipdnn_frontend::graph::DataType_t::FLOAT);
    EXPECT_EQ(input_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(input_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(output_tensor->get_uid(), 2);
    EXPECT_EQ(output_tensor->get_name(), "OutputTensor");
    EXPECT_EQ(output_tensor->get_data_type(), hipdnn_frontend::graph::DataType_t::HALF);
    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{5, 6, 7, 8}));
    EXPECT_EQ(output_tensor->get_stride(), (std::vector<int64_t>{1, 2, 3, 4}));

    EXPECT_EQ(pointwise_attributes.get_operation(), hipdnn_frontend::graph::PointwiseMode_t::RELU);
    EXPECT_EQ(pointwise_attributes.get_relu_lower_clip(), 0.1f);
    EXPECT_EQ(pointwise_attributes.get_relu_upper_clip(), 6.0f);
    EXPECT_EQ(pointwise_attributes.get_relu_lower_slope(), 0.01f);
    EXPECT_EQ(pointwise_attributes.get_axis(), 1);
}
TEST(FrontendAPITests, CreateBatchnormInferenceAttributes)
{
    hipdnn_frontend::graph::Batchnorm_inference_attributes batchnorm_attributes;

    batchnorm_attributes.set_x(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_y(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_mean(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_inv_variance(
        std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_bias(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());

    auto input_tensor = batchnorm_attributes.get_x();
    input_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = batchnorm_attributes.get_y();
    output_tensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto mean_tensor = batchnorm_attributes.get_mean();
    mean_tensor->set_uid(3)
        .set_name("MeanTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto variance_tensor = batchnorm_attributes.get_inv_variance();
    variance_tensor->set_uid(4)
        .set_name("VarianceTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto scale_tensor = batchnorm_attributes.get_scale();
    scale_tensor->set_uid(5)
        .set_name("ScaleTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto bias_tensor = batchnorm_attributes.get_bias();
    bias_tensor->set_uid(6)
        .set_name("BiasTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    EXPECT_EQ(input_tensor->get_uid(), 1);
    EXPECT_EQ(input_tensor->get_name(), "InputTensor");
    EXPECT_EQ(input_tensor->get_data_type(), hipdnn_frontend::graph::DataType_t::FLOAT);
    EXPECT_EQ(input_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(input_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(output_tensor->get_uid(), 2);
    EXPECT_EQ(output_tensor->get_name(), "OutputTensor");
    EXPECT_EQ(output_tensor->get_data_type(), hipdnn_frontend::graph::DataType_t::FLOAT);
    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(output_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(mean_tensor->get_uid(), 3);
    EXPECT_EQ(mean_tensor->get_name(), "MeanTensor");
    EXPECT_EQ(mean_tensor->get_data_type(), hipdnn_frontend::graph::DataType_t::FLOAT);
    EXPECT_EQ(mean_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(mean_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(variance_tensor->get_uid(), 4);
    EXPECT_EQ(variance_tensor->get_name(), "VarianceTensor");
    EXPECT_EQ(variance_tensor->get_data_type(), hipdnn_frontend::graph::DataType_t::FLOAT);
    EXPECT_EQ(variance_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(variance_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(scale_tensor->get_uid(), 5);
    EXPECT_EQ(scale_tensor->get_name(), "ScaleTensor");
    EXPECT_EQ(scale_tensor->get_data_type(), hipdnn_frontend::graph::DataType_t::FLOAT);
    EXPECT_EQ(scale_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(scale_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(bias_tensor->get_uid(), 6);
    EXPECT_EQ(bias_tensor->get_name(), "BiasTensor");
    EXPECT_EQ(bias_tensor->get_data_type(), hipdnn_frontend::graph::DataType_t::FLOAT);
    EXPECT_EQ(bias_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(bias_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}
