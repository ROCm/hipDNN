// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(PointwiseNodeTests, SingleInput)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto inputTensor = attributes.get_input_0();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(2).set_name("OutputTensor");

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK);

    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(PointwiseNodeTests, TwoInputs)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3).set_name("OutputTensor");

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK);

    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(PointwiseNodeTests, ThreeInputs)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_input_2(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto inputTensor2 = attributes.get_input_2();
    inputTensor2->set_uid(3)
        .set_name("InputTensor2")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(4).set_name("OutputTensor");

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK);

    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(PointwiseNodeTests, PreValidateNode)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode_t::RELU_FWD);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK);
}

TEST(PointwiseNodeTests, PreValidateNodeMissingValues)
{
    PointwiseAttributes attributes;

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    attributes.set_input_0(std::make_shared<TensorAttributes>());
    auto attributesCopy = attributes;
    PointwiseNode nodeWithInput(std::move(attributesCopy), graphAttributes);

    error = nodeWithInput.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributesCopy = attributes;
    PointwiseNode nodeWithOutput(std::move(attributesCopy), graphAttributes);

    error = nodeWithOutput.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    attributes.set_mode(PointwiseMode_t::RELU_FWD);
    attributesCopy = attributes;
    PointwiseNode nodeWithAllValues(std::move(attributesCopy), graphAttributes);

    error = nodeWithAllValues.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK);
}

TEST(PointwiseNodeTests, InferPropertiesNode)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto inputTensor = attributes.get_input_0();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(2).set_name("OutputTensor");

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK);

    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(PointwiseNodeTests, PackNode)
{
    PointwiseAttributes attributes;
    attributes.name = "PointwiseNode";

    auto inputTensor = std::make_shared<TensorAttributes>();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    attributes.set_input_0(inputTensor);

    auto outputTensor = std::make_shared<TensorAttributes>();
    outputTensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    attributes.set_output_0(outputTensor);

    attributes.set_mode(PointwiseMode_t::RELU_FWD);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto bufferPointer = builder.GetBufferPointer();
    auto nodeFlatbuffer = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Node>(bufferPointer);

    EXPECT_STREQ(nodeFlatbuffer->name()->c_str(), "PointwiseNode");
    EXPECT_EQ(nodeFlatbuffer->attributes_type(),
              hipdnn_sdk::data_objects::NodeAttributes_PointwiseAttributes);

    auto packedAttributes = nodeFlatbuffer->attributes_as_PointwiseAttributes();
    ASSERT_NE(packedAttributes, nullptr);

    EXPECT_EQ(packedAttributes->in_0_tensor_uid(), inputTensor->get_uid());
    EXPECT_EQ(packedAttributes->out_0_tensor_uid(), outputTensor->get_uid());
    EXPECT_EQ(packedAttributes->operation(), static_cast<int>(PointwiseMode_t::RELU_FWD));
}
