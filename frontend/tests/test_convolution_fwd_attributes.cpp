// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/convolution_fwd_attributes.hpp>

TEST(ConvolutionFwdAttributesTests, CreateConvolutionFwdAttributes)
{
    hipdnn_frontend::graph::Conv_fprop_attributes conv_attributes;

    // Set tensors
    conv_attributes.set_x(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    conv_attributes.set_w(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    conv_attributes.set_y(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());

    // Set convolution parameters
    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});
    conv_attributes.set_conv_mode(hipdnn_frontend::ConvolutionMode_t::CROSS_CORRELATION);

    // Configure input tensor
    auto x_tensor = conv_attributes.get_x();
    x_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 3, 32, 32}) // NCHW format
        .set_stride({3072, 1024, 32, 1});

    // Configure weights tensor
    auto w_tensor = conv_attributes.get_w();
    w_tensor->set_uid(2)
        .set_name("WeightsTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({64, 3, 3, 3}) // KCHW format
        .set_stride({27, 9, 3, 1});

    // Configure output tensor
    auto y_tensor = conv_attributes.get_y();
    y_tensor->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 64, 32, 32}) // NCHW format
        .set_stride({65536, 1024, 32, 1});

    // Verify tensor attributes
    EXPECT_EQ(x_tensor->get_uid(), 1);
    EXPECT_EQ(x_tensor->get_name(), "InputTensor");
    EXPECT_EQ(x_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(x_tensor->get_dim(), (std::vector<int64_t>{1, 3, 32, 32}));
    EXPECT_EQ(x_tensor->get_stride(), (std::vector<int64_t>{3072, 1024, 32, 1}));

    EXPECT_EQ(w_tensor->get_uid(), 2);
    EXPECT_EQ(w_tensor->get_name(), "WeightsTensor");
    EXPECT_EQ(w_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(w_tensor->get_dim(), (std::vector<int64_t>{64, 3, 3, 3}));
    EXPECT_EQ(w_tensor->get_stride(), (std::vector<int64_t>{27, 9, 3, 1}));

    EXPECT_EQ(y_tensor->get_uid(), 3);
    EXPECT_EQ(y_tensor->get_name(), "OutputTensor");
    EXPECT_EQ(y_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(y_tensor->get_dim(), (std::vector<int64_t>{1, 64, 32, 32}));
    EXPECT_EQ(y_tensor->get_stride(), (std::vector<int64_t>{65536, 1024, 32, 1}));

    // Verify convolution parameters
    EXPECT_EQ(conv_attributes.get_pre_padding(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(conv_attributes.get_post_padding(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(conv_attributes.get_stride(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(conv_attributes.get_dilation(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(conv_attributes.get_conv_mode(),
              hipdnn_frontend::ConvolutionMode_t::CROSS_CORRELATION);
}

TEST(ConvolutionFwdAttributesTests, PackAttributes)
{
    hipdnn_frontend::graph::Conv_fprop_attributes conv_attributes;

    // Set tensors
    auto x_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    x_tensor->set_uid(1);
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    w_tensor->set_uid(2);
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    y_tensor->set_uid(3);
    conv_attributes.set_y(y_tensor);

    // Set convolution parameters
    conv_attributes.set_pre_padding({2, 2});
    conv_attributes.set_post_padding({2, 2});
    conv_attributes.set_stride({2, 2});
    conv_attributes.set_dilation({1, 1});
    conv_attributes.set_conv_mode(hipdnn_frontend::ConvolutionMode_t::CROSS_CORRELATION);

    // Pack attributes
    flatbuffers::FlatBufferBuilder builder;
    auto packed_attributes = conv_attributes.pack_attributes(builder);
    builder.Finish(packed_attributes);

    auto buffer = builder.GetBufferPointer();
    auto conv_attributes_fb
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::ConvolutionFwdAttributes>(buffer);

    // Verify packed tensor UIDs
    EXPECT_EQ(conv_attributes_fb->x_tensor_uid(), 1);
    EXPECT_EQ(conv_attributes_fb->w_tensor_uid(), 2);
    EXPECT_EQ(conv_attributes_fb->y_tensor_uid(), 3);

    // Verify packed convolution parameters
    ASSERT_EQ(conv_attributes_fb->pre_padding()->size(), 2);
    EXPECT_EQ(conv_attributes_fb->pre_padding()->Get(0), 2);
    EXPECT_EQ(conv_attributes_fb->pre_padding()->Get(1), 2);

    ASSERT_EQ(conv_attributes_fb->post_padding()->size(), 2);
    EXPECT_EQ(conv_attributes_fb->post_padding()->Get(0), 2);
    EXPECT_EQ(conv_attributes_fb->post_padding()->Get(1), 2);

    ASSERT_EQ(conv_attributes_fb->stride()->size(), 2);
    EXPECT_EQ(conv_attributes_fb->stride()->Get(0), 2);
    EXPECT_EQ(conv_attributes_fb->stride()->Get(1), 2);

    ASSERT_EQ(conv_attributes_fb->dilation()->size(), 2);
    EXPECT_EQ(conv_attributes_fb->dilation()->Get(0), 1);
    EXPECT_EQ(conv_attributes_fb->dilation()->Get(1), 1);

    EXPECT_EQ(conv_attributes_fb->conv_mode(),
              hipdnn_sdk::data_objects::ConvMode_CROSS_CORRELATION);
}

TEST(ConvolutionFwdAttributesTests, DefaultValues)
{
    hipdnn_frontend::graph::Conv_fprop_attributes conv_attributes;

    // Check default convolution mode
    EXPECT_EQ(conv_attributes.get_conv_mode(),
              hipdnn_frontend::ConvolutionMode_t::CROSS_CORRELATION);

    // Check that parameters are empty by default
    EXPECT_TRUE(conv_attributes.get_pre_padding().empty());
    EXPECT_TRUE(conv_attributes.get_post_padding().empty());
    EXPECT_TRUE(conv_attributes.get_stride().empty());
    EXPECT_TRUE(conv_attributes.get_dilation().empty());

    // Check that tensors are null by default
    EXPECT_EQ(conv_attributes.get_x(), nullptr);
    EXPECT_EQ(conv_attributes.get_w(), nullptr);
    EXPECT_EQ(conv_attributes.get_y(), nullptr);
}

TEST(ConvolutionFwdAttributesTests, SetPrePaddingWithMove)
{
    hipdnn_frontend::graph::Conv_fprop_attributes conv_attributes;

    std::vector<int64_t> pre_padding_move = {3, 3};
    conv_attributes.set_pre_padding(std::move(pre_padding_move));

    EXPECT_EQ(conv_attributes.get_pre_padding(), (std::vector<int64_t>{3, 3}));
}

TEST(ConvolutionFwdAttributesTests, SetPostPaddingWithMove)
{
    hipdnn_frontend::graph::Conv_fprop_attributes conv_attributes;

    std::vector<int64_t> post_padding_move = {4, 4};
    conv_attributes.set_post_padding(std::move(post_padding_move));

    EXPECT_EQ(conv_attributes.get_post_padding(), (std::vector<int64_t>{4, 4}));
}

TEST(ConvolutionFwdAttributesTests, SetStrideWithMove)
{
    hipdnn_frontend::graph::Conv_fprop_attributes conv_attributes;

    std::vector<int64_t> stride_move = {2, 2};
    conv_attributes.set_stride(std::move(stride_move));

    EXPECT_EQ(conv_attributes.get_stride(), (std::vector<int64_t>{2, 2}));
}

TEST(ConvolutionFwdAttributesTests, SetDilationWithMove)
{
    hipdnn_frontend::graph::Conv_fprop_attributes conv_attributes;

    std::vector<int64_t> dilation_move = {2, 2};
    conv_attributes.set_dilation(std::move(dilation_move));

    EXPECT_EQ(conv_attributes.get_dilation(), (std::vector<int64_t>{2, 2}));
}

TEST(ConvolutionFwdAttributesTests, SetPaddingBothPreAndPost)
{
    hipdnn_frontend::graph::Conv_fprop_attributes conv_attributes;

    std::vector<int64_t> padding = {5, 5};
    conv_attributes.set_padding(padding);

    // set_padding should set both pre and post padding to the same value
    EXPECT_EQ(conv_attributes.get_pre_padding(), (std::vector<int64_t>{5, 5}));
    EXPECT_EQ(conv_attributes.get_post_padding(), (std::vector<int64_t>{5, 5}));
}

TEST(ConvolutionFwdAttributesTests, SetXMove)
{
    hipdnn_frontend::graph::Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    x_tensor->set_uid(10)
        .set_name("MovedInputTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 3, 224, 224})
        .set_stride({150528, 50176, 224, 1});

    // Store the raw pointer before moving
    auto raw_ptr = x_tensor.get();

    conv_attributes.set_x(std::move(x_tensor));

    // After move, x_tensor should be nullptr
    EXPECT_EQ(x_tensor, nullptr);

    // The moved tensor should be accessible through get_x()
    auto retrieved_tensor = conv_attributes.get_x();
    EXPECT_EQ(retrieved_tensor.get(), raw_ptr);
}

TEST(ConvolutionFwdAttributesTests, SetWMove)
{
    hipdnn_frontend::graph::Conv_fprop_attributes conv_attributes;

    auto w_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    w_tensor->set_uid(20)
        .set_name("MovedWeightTensor")
        .set_data_type(hipdnn_frontend::DataType_t::HALF)
        .set_dim({128, 64, 5, 5})
        .set_stride({1600, 25, 5, 1});

    // Store the raw pointer before moving
    auto raw_ptr = w_tensor.get();

    conv_attributes.set_w(std::move(w_tensor));

    // After move, w_tensor should be nullptr
    EXPECT_EQ(w_tensor, nullptr);

    // The moved tensor should be accessible through get_w()
    auto retrieved_tensor = conv_attributes.get_w();
    EXPECT_EQ(retrieved_tensor.get(), raw_ptr);
}

TEST(ConvolutionFwdAttributesTests, SetYWithMove)
{
    hipdnn_frontend::graph::Conv_fprop_attributes conv_attributes;

    auto y_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    y_tensor->set_uid(30)
        .set_name("MovedOutputTensor")
        .set_data_type(hipdnn_frontend::DataType_t::BFLOAT16)
        .set_dim({1, 128, 112, 112})
        .set_stride({1605632, 12544, 112, 1});

    // Store the raw pointer before moving
    auto raw_ptr = y_tensor.get();

    conv_attributes.set_y(std::move(y_tensor));

    // After move, y_tensor should be nullptr
    EXPECT_EQ(y_tensor, nullptr);

    // The moved tensor should be accessible through get_y()
    auto retrieved_tensor = conv_attributes.get_y();
    EXPECT_EQ(retrieved_tensor.get(), raw_ptr);
}

TEST(ConvolutionFwdAttributesTests, SetTensorsConstRef)
{
    hipdnn_frontend::graph::Conv_fprop_attributes conv_attributes;

    // Create tensors
    auto x_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    x_tensor->set_uid(40).set_name("InputConstRef");

    auto w_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    w_tensor->set_uid(50).set_name("WeightConstRef");

    auto y_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    y_tensor->set_uid(60).set_name("OutputConstRef");

    // Set using const reference (copy)
    conv_attributes.set_x(x_tensor);
    conv_attributes.set_w(w_tensor);
    conv_attributes.set_y(y_tensor);

    // Original tensors should still be valid
    EXPECT_NE(x_tensor, nullptr);
    EXPECT_NE(w_tensor, nullptr);
    EXPECT_NE(y_tensor, nullptr);
}
