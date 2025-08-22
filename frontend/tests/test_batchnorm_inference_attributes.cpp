// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>

TEST(BatchnormInferenceAttributesTests, CreateBatchnormInferenceAttributes)
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
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = batchnorm_attributes.get_y();
    output_tensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto mean_tensor = batchnorm_attributes.get_mean();
    mean_tensor->set_uid(3)
        .set_name("MeanTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto variance_tensor = batchnorm_attributes.get_inv_variance();
    variance_tensor->set_uid(4)
        .set_name("VarianceTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto scale_tensor = batchnorm_attributes.get_scale();
    scale_tensor->set_uid(5)
        .set_name("ScaleTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto bias_tensor = batchnorm_attributes.get_bias();
    bias_tensor->set_uid(6)
        .set_name("BiasTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    EXPECT_EQ(input_tensor->get_uid(), 1);
    EXPECT_EQ(input_tensor->get_name(), "InputTensor");
    EXPECT_EQ(input_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(input_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(input_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(output_tensor->get_uid(), 2);
    EXPECT_EQ(output_tensor->get_name(), "OutputTensor");
    EXPECT_EQ(output_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(output_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(mean_tensor->get_uid(), 3);
    EXPECT_EQ(mean_tensor->get_name(), "MeanTensor");
    EXPECT_EQ(mean_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(mean_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(mean_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(variance_tensor->get_uid(), 4);
    EXPECT_EQ(variance_tensor->get_name(), "VarianceTensor");
    EXPECT_EQ(variance_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(variance_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(variance_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(scale_tensor->get_uid(), 5);
    EXPECT_EQ(scale_tensor->get_name(), "ScaleTensor");
    EXPECT_EQ(scale_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(scale_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(scale_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(bias_tensor->get_uid(), 6);
    EXPECT_EQ(bias_tensor->get_name(), "BiasTensor");
    EXPECT_EQ(bias_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(bias_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(bias_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(BatchnormInferenceAttributesTests, SetXWithMove)
{
    hipdnn_frontend::graph::Batchnorm_inference_attributes batchnorm_attributes;

    auto x_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    x_tensor->set_uid(1).set_name("XTensor");

    auto raw_ptr = x_tensor.get();

    batchnorm_attributes.set_x(std::move(x_tensor));

    auto retrieved = batchnorm_attributes.get_x();
    EXPECT_EQ(retrieved->get_uid(), 1);
    EXPECT_EQ(retrieved->get_name(), "XTensor");

    EXPECT_EQ(x_tensor, nullptr);
    EXPECT_EQ(retrieved.get(), raw_ptr);
}

TEST(BatchnormInferenceAttributesTests, SetMeanWithMove)
{
    hipdnn_frontend::graph::Batchnorm_inference_attributes batchnorm_attributes;

    auto mean_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    mean_tensor->set_uid(2).set_name("MeanTensor");

    auto raw_ptr = mean_tensor.get();

    batchnorm_attributes.set_mean(std::move(mean_tensor));

    auto retrieved = batchnorm_attributes.get_mean();
    EXPECT_EQ(retrieved->get_uid(), 2);
    EXPECT_EQ(retrieved->get_name(), "MeanTensor");

    EXPECT_EQ(mean_tensor, nullptr);
    EXPECT_EQ(retrieved.get(), raw_ptr);
}

TEST(BatchnormInferenceAttributesTests, SetInvVarianceWithMove)
{
    hipdnn_frontend::graph::Batchnorm_inference_attributes batchnorm_attributes;

    auto inv_variance_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    inv_variance_tensor->set_uid(3).set_name("InvVarianceTensor");

    auto raw_ptr = inv_variance_tensor.get();

    batchnorm_attributes.set_inv_variance(std::move(inv_variance_tensor));

    auto retrieved = batchnorm_attributes.get_inv_variance();
    EXPECT_EQ(retrieved->get_uid(), 3);
    EXPECT_EQ(retrieved->get_name(), "InvVarianceTensor");

    EXPECT_EQ(inv_variance_tensor, nullptr);
    EXPECT_EQ(retrieved.get(), raw_ptr);
}

TEST(BatchnormInferenceAttributesTests, SetScaleWithMove)
{
    hipdnn_frontend::graph::Batchnorm_inference_attributes batchnorm_attributes;

    auto scale_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    scale_tensor->set_uid(4).set_name("ScaleTensor");

    auto raw_ptr = scale_tensor.get();

    batchnorm_attributes.set_scale(std::move(scale_tensor));

    auto retrieved = batchnorm_attributes.get_scale();
    EXPECT_EQ(retrieved->get_uid(), 4);
    EXPECT_EQ(retrieved->get_name(), "ScaleTensor");

    EXPECT_EQ(scale_tensor, nullptr);
    EXPECT_EQ(retrieved.get(), raw_ptr);
}

TEST(BatchnormInferenceAttributesTests, SetBiasWithMove)
{
    hipdnn_frontend::graph::Batchnorm_inference_attributes batchnorm_attributes;

    auto bias_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    bias_tensor->set_uid(5).set_name("BiasTensor");

    auto raw_ptr = bias_tensor.get();

    batchnorm_attributes.set_bias(std::move(bias_tensor));

    auto retrieved = batchnorm_attributes.get_bias();
    EXPECT_EQ(retrieved->get_uid(), 5);
    EXPECT_EQ(retrieved->get_name(), "BiasTensor");

    EXPECT_EQ(bias_tensor, nullptr);
    EXPECT_EQ(retrieved.get(), raw_ptr);
}

TEST(BatchnormInferenceAttributesTests, SetYWithMove)
{
    hipdnn_frontend::graph::Batchnorm_inference_attributes batchnorm_attributes;

    auto y_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    y_tensor->set_uid(6).set_name("YTensor");

    auto raw_ptr = y_tensor.get();

    batchnorm_attributes.set_y(std::move(y_tensor));

    auto retrieved = batchnorm_attributes.get_y();
    EXPECT_EQ(retrieved->get_uid(), 6);
    EXPECT_EQ(retrieved->get_name(), "YTensor");

    EXPECT_EQ(y_tensor, nullptr);
    EXPECT_EQ(retrieved.get(), raw_ptr);
}

// Simplified move tests - testing move semantics without setting uid/name

TEST(BatchnormInferenceAttributesTests, SimplifiedSetXWithMove)
{
    hipdnn_frontend::graph::Batchnorm_inference_attributes batchnorm_attributes;

    auto x_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    batchnorm_attributes.set_x(std::move(x_tensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnorm_attributes.get_x(), nullptr);
}

TEST(BatchnormInferenceAttributesTests, SimplifiedSetMeanWithMove)
{
    hipdnn_frontend::graph::Batchnorm_inference_attributes batchnorm_attributes;

    auto mean_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    batchnorm_attributes.set_mean(std::move(mean_tensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnorm_attributes.get_mean(), nullptr);
}

TEST(BatchnormInferenceAttributesTests, SimplifiedSetInvVarianceWithMove)
{
    hipdnn_frontend::graph::Batchnorm_inference_attributes batchnorm_attributes;

    auto inv_variance_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    batchnorm_attributes.set_inv_variance(std::move(inv_variance_tensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnorm_attributes.get_inv_variance(), nullptr);
}

TEST(BatchnormInferenceAttributesTests, SimplifiedSetScaleWithMove)
{
    hipdnn_frontend::graph::Batchnorm_inference_attributes batchnorm_attributes;

    auto scale_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    batchnorm_attributes.set_scale(std::move(scale_tensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnorm_attributes.get_scale(), nullptr);
}

TEST(BatchnormInferenceAttributesTests, SimplifiedSetBiasWithMove)
{
    hipdnn_frontend::graph::Batchnorm_inference_attributes batchnorm_attributes;

    auto bias_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    batchnorm_attributes.set_bias(std::move(bias_tensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnorm_attributes.get_bias(), nullptr);
}

TEST(BatchnormInferenceAttributesTests, SimplifiedSetYWithMove)
{
    hipdnn_frontend::graph::Batchnorm_inference_attributes batchnorm_attributes;

    auto y_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    batchnorm_attributes.set_y(std::move(y_tensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnorm_attributes.get_y(), nullptr);
}
