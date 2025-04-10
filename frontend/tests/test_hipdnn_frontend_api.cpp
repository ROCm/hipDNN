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

    auto deserializedGraph = flatbuffers::GetRoot<Graph>(builder.GetBufferPointer());
    EXPECT_EQ(deserializedGraph->name()->str(), "Graph");
}

TEST(FrontendAPITests, CreatePointwiseAttributes)
{
    hipdnn_frontend::graph::Pointwise_Attributes pointwiseAttributes;

    pointwiseAttributes.set_input_0(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    pointwiseAttributes.set_output_0(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    pointwiseAttributes.set_operation(hipdnn_frontend::graph::PointwiseMode_t::RELU)
        .set_relu_lower_clip(0.1f)
        .set_relu_upper_clip(6.0f)
        .set_relu_lower_slope(0.01f)
        .set_axis(1);

    auto inputTensor = pointwiseAttributes.get_input_0();
    EXPECT_FALSE(inputTensor->has_uid());

    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = pointwiseAttributes.get_output_0();
    EXPECT_FALSE(outputTensor->has_uid());

    outputTensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::HALF)
        .set_dim({5, 6, 7, 8})
        .set_stride({1, 2, 3, 4});

    EXPECT_EQ(inputTensor->get_uid(), 1);
    EXPECT_EQ(inputTensor->get_name(), "InputTensor");
    EXPECT_EQ(inputTensor->get_data_type(), hipdnn_frontend::graph::DataType_t::FLOAT);
    EXPECT_EQ(inputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(inputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(outputTensor->get_uid(), 2);
    EXPECT_EQ(outputTensor->get_name(), "OutputTensor");
    EXPECT_EQ(outputTensor->get_data_type(), hipdnn_frontend::graph::DataType_t::HALF);
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{5, 6, 7, 8}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{1, 2, 3, 4}));

    EXPECT_EQ(pointwiseAttributes.get_operation(), hipdnn_frontend::graph::PointwiseMode_t::RELU);
    EXPECT_EQ(pointwiseAttributes.get_relu_lower_clip(), 0.1f);
    EXPECT_EQ(pointwiseAttributes.get_relu_upper_clip(), 6.0f);
    EXPECT_EQ(pointwiseAttributes.get_relu_lower_slope(), 0.01f);
    EXPECT_EQ(pointwiseAttributes.get_axis(), 1);
}
TEST(FrontendAPITests, CreateBatchnormInferenceAttributes)
{
    hipdnn_frontend::graph::Batchnorm_inference_attributes batchnormAttributes;

    batchnormAttributes.set_x(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnormAttributes.set_y(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnormAttributes.set_mean(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnormAttributes.set_inv_variance(
        std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnormAttributes.set_scale(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnormAttributes.set_bias(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());

    auto inputTensor = batchnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = batchnormAttributes.get_y();
    outputTensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto meanTensor = batchnormAttributes.get_mean();
    meanTensor->set_uid(3)
        .set_name("MeanTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto varianceTensor = batchnormAttributes.get_inv_variance();
    varianceTensor->set_uid(4)
        .set_name("VarianceTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto scaleTensor = batchnormAttributes.get_scale();
    scaleTensor->set_uid(5)
        .set_name("ScaleTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto biasTensor = batchnormAttributes.get_bias();
    biasTensor->set_uid(6)
        .set_name("BiasTensor")
        .set_data_type(hipdnn_frontend::graph::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    EXPECT_EQ(inputTensor->get_uid(), 1);
    EXPECT_EQ(inputTensor->get_name(), "InputTensor");
    EXPECT_EQ(inputTensor->get_data_type(), hipdnn_frontend::graph::DataType_t::FLOAT);
    EXPECT_EQ(inputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(inputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(outputTensor->get_uid(), 2);
    EXPECT_EQ(outputTensor->get_name(), "OutputTensor");
    EXPECT_EQ(outputTensor->get_data_type(), hipdnn_frontend::graph::DataType_t::FLOAT);
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(meanTensor->get_uid(), 3);
    EXPECT_EQ(meanTensor->get_name(), "MeanTensor");
    EXPECT_EQ(meanTensor->get_data_type(), hipdnn_frontend::graph::DataType_t::FLOAT);
    EXPECT_EQ(meanTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(meanTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(varianceTensor->get_uid(), 4);
    EXPECT_EQ(varianceTensor->get_name(), "VarianceTensor");
    EXPECT_EQ(varianceTensor->get_data_type(), hipdnn_frontend::graph::DataType_t::FLOAT);
    EXPECT_EQ(varianceTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(varianceTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(scaleTensor->get_uid(), 5);
    EXPECT_EQ(scaleTensor->get_name(), "ScaleTensor");
    EXPECT_EQ(scaleTensor->get_data_type(), hipdnn_frontend::graph::DataType_t::FLOAT);
    EXPECT_EQ(scaleTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(scaleTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(biasTensor->get_uid(), 6);
    EXPECT_EQ(biasTensor->get_name(), "BiasTensor");
    EXPECT_EQ(biasTensor->get_data_type(), hipdnn_frontend::graph::DataType_t::FLOAT);
    EXPECT_EQ(biasTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(biasTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}
