// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/pointwise_attributes.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(PointwiseAttributesTests, CreatePointwiseAttributes)
{
    Pointwise_attributes pointwise_attributes;

    pointwise_attributes.set_input_0(std::make_shared<Tensor_attributes>());
    pointwise_attributes.set_output_0(std::make_shared<Tensor_attributes>());
    pointwise_attributes.set_mode(PointwiseMode_t::RELU_FWD)
        .set_relu_lower_clip(0.1f)
        .set_relu_upper_clip(6.0f)
        .set_relu_lower_clip_slope(0.01f)
        .set_axis(1);

    auto input_tensor = pointwise_attributes.get_input_0();
    EXPECT_FALSE(input_tensor->has_uid());

    input_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = pointwise_attributes.get_output_0();
    EXPECT_FALSE(output_tensor->has_uid());

    output_tensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(DataType_t::HALF)
        .set_dim({5, 6, 7, 8})
        .set_stride({1, 2, 3, 4});

    EXPECT_EQ(input_tensor->get_uid(), 1);
    EXPECT_EQ(input_tensor->get_name(), "InputTensor");
    EXPECT_EQ(input_tensor->get_data_type(), DataType_t::FLOAT);
    EXPECT_EQ(input_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(input_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(output_tensor->get_uid(), 2);
    EXPECT_EQ(output_tensor->get_name(), "OutputTensor");
    EXPECT_EQ(output_tensor->get_data_type(), DataType_t::HALF);
    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{5, 6, 7, 8}));
    EXPECT_EQ(output_tensor->get_stride(), (std::vector<int64_t>{1, 2, 3, 4}));

    EXPECT_EQ(pointwise_attributes.get_mode(), PointwiseMode_t::RELU_FWD);
    EXPECT_EQ(pointwise_attributes.get_relu_lower_clip(), 0.1f);
    EXPECT_EQ(pointwise_attributes.get_relu_upper_clip(), 6.0f);
    EXPECT_EQ(pointwise_attributes.get_relu_lower_slope(), 0.01f);
    EXPECT_EQ(pointwise_attributes.get_axis(), 1);
}

TEST(PointwiseAttributesTests, CreatePointwiseAttributesWithTwoInputs)
{
    Pointwise_attributes pointwise_attributes;

    pointwise_attributes.set_input_0(std::make_shared<Tensor_attributes>());
    pointwise_attributes.set_input_1(std::make_shared<Tensor_attributes>());
    pointwise_attributes.set_output_0(std::make_shared<Tensor_attributes>());
    pointwise_attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto input_tensor_0 = pointwise_attributes.get_input_0();
    input_tensor_0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto input_tensor_1 = pointwise_attributes.get_input_1();
    input_tensor_1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = pointwise_attributes.get_output_0();
    output_tensor->set_uid(3).set_name("OutputTensor");

    EXPECT_EQ(input_tensor_0->get_uid(), 1);
    EXPECT_EQ(input_tensor_0->get_name(), "InputTensor0");
    EXPECT_EQ(input_tensor_1->get_uid(), 2);
    EXPECT_EQ(input_tensor_1->get_name(), "InputTensor1");
    EXPECT_EQ(output_tensor->get_uid(), 3);
    EXPECT_EQ(output_tensor->get_name(), "OutputTensor");
    EXPECT_EQ(pointwise_attributes.get_mode(), PointwiseMode_t::RELU_FWD);
}

TEST(PointwiseAttributesTests, CreatePointwiseAttributesWithThreeInputs)
{
    Pointwise_attributes pointwise_attributes;

    pointwise_attributes.set_input_0(std::make_shared<Tensor_attributes>());
    pointwise_attributes.set_input_1(std::make_shared<Tensor_attributes>());
    pointwise_attributes.set_input_2(std::make_shared<Tensor_attributes>());
    pointwise_attributes.set_output_0(std::make_shared<Tensor_attributes>());
    pointwise_attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto input_tensor_0 = pointwise_attributes.get_input_0();
    input_tensor_0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto input_tensor_1 = pointwise_attributes.get_input_1();
    input_tensor_1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto input_tensor_2 = pointwise_attributes.get_input_2();
    input_tensor_2->set_uid(3)
        .set_name("InputTensor2")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = pointwise_attributes.get_output_0();
    output_tensor->set_uid(4).set_name("OutputTensor");

    EXPECT_EQ(input_tensor_0->get_uid(), 1);
    EXPECT_EQ(input_tensor_0->get_name(), "InputTensor0");
    EXPECT_EQ(input_tensor_1->get_uid(), 2);
    EXPECT_EQ(input_tensor_1->get_name(), "InputTensor1");
    EXPECT_EQ(input_tensor_2->get_uid(), 3);
    EXPECT_EQ(input_tensor_2->get_name(), "InputTensor2");
    EXPECT_EQ(output_tensor->get_uid(), 4);
    EXPECT_EQ(output_tensor->get_name(), "OutputTensor");
    EXPECT_EQ(pointwise_attributes.get_mode(), PointwiseMode_t::RELU_FWD);
}