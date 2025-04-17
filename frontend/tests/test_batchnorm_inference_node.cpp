#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_frontend/node/batchnorm_inference_node.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(BatchnormInferenceNodeTests, BatchnormInferenceNodeProperties)
{
    Batchnorm_inference_attributes batchnorm_attributes;
    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_y(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_mean(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_inv_variance(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_bias(std::make_shared<Tensor_attributes>());

    auto input_tensor = batchnorm_attributes.get_x();
    input_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = batchnorm_attributes.get_y();
    output_tensor->set_uid(2).set_name("OutputTensor");

    Graph_attributes       graph_attributes;
    BatchnormInferenceNode node(std::move(batchnorm_attributes), graph_attributes);
    auto                   error = node.infer_properties_node();

    EXPECT_EQ(error.code, error_code_t::OK);
    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(output_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(BatchnormInferenceNodeTests, PreValidateNode)
{
    Batchnorm_inference_attributes batchnorm_attributes;
    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_y(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_bias(std::make_shared<Tensor_attributes>());

    Graph_attributes       graph_attributes;
    BatchnormInferenceNode node(std::move(batchnorm_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK);
}

TEST(BatchnormInferenceNodeTests, PreValidateNodeMissingValues)
{
    Batchnorm_inference_attributes batchnorm_attributes;

    Graph_attributes       graph_attributes;
    BatchnormInferenceNode node(std::move(batchnorm_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    auto                   batchnorm_attributes_copy = batchnorm_attributes;
    BatchnormInferenceNode node_with_x(std::move(batchnorm_attributes_copy), graph_attributes);

    error = node_with_x.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_y(std::make_shared<Tensor_attributes>());
    batchnorm_attributes_copy = batchnorm_attributes;
    BatchnormInferenceNode node_with_y(std::move(batchnorm_attributes_copy), graph_attributes);

    error = node_with_y.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes_copy = batchnorm_attributes;
    BatchnormInferenceNode node_with_scale(std::move(batchnorm_attributes_copy), graph_attributes);

    error = node_with_scale.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_bias(std::make_shared<Tensor_attributes>());
    batchnorm_attributes_copy = batchnorm_attributes;
    BatchnormInferenceNode node_with_all_values(std::move(batchnorm_attributes_copy),
                                                graph_attributes);

    error = node_with_all_values.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK);
}

TEST(BatchnormInferenceNodeTests, InferPropertiesNode)
{
    Batchnorm_inference_attributes batchnorm_attributes;
    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_y(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_mean(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_inv_variance(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_bias(std::make_shared<Tensor_attributes>());

    auto input_tensor = batchnorm_attributes.get_x();
    input_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = batchnorm_attributes.get_y();
    output_tensor->set_uid(2).set_name("OutputTensor");

    Graph_attributes       graph_attributes;
    BatchnormInferenceNode node(std::move(batchnorm_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK);

    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(output_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(BatchnormInferenceNodeTests, PackNode)
{
    Batchnorm_inference_attributes batchnorm_attributes;
    batchnorm_attributes.name = "BatchnormInference";

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_uid(1)
        .set_name("XTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnorm_attributes.set_x(x_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_uid(2)
        .set_name("YTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnorm_attributes.set_y(y_tensor);

    auto mean_tensor = std::make_shared<Tensor_attributes>();
    mean_tensor->set_uid(3)
        .set_name("MeanTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnorm_attributes.set_mean(mean_tensor);

    auto inv_variance_tensor = std::make_shared<Tensor_attributes>();
    inv_variance_tensor->set_uid(4)
        .set_name("InvVarianceTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnorm_attributes.set_inv_variance(inv_variance_tensor);

    auto scale_tensor = std::make_shared<Tensor_attributes>();
    scale_tensor->set_uid(5)
        .set_name("ScaleTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnorm_attributes.set_scale(scale_tensor);

    auto bias_tensor = std::make_shared<Tensor_attributes>();
    bias_tensor->set_uid(6)
        .set_name("BiasTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnorm_attributes.set_bias(bias_tensor);

    Graph_attributes       graph_attributes;
    BatchnormInferenceNode node(std::move(batchnorm_attributes), graph_attributes);

    flatbuffers::FlatBufferBuilder builder;
    auto                           offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto buffer_pointer  = builder.GetBufferPointer();
    auto node_flatbuffer = flatbuffers::GetRoot<hipdnn::sdk::Node>(buffer_pointer);

    EXPECT_STREQ(node_flatbuffer->name()->c_str(), "BatchnormInference");
    EXPECT_EQ(node_flatbuffer->attributes_type(),
              hipdnn::sdk::NodeAttributes_BatchnormInferenceAttributes);

    auto packed_attributes = node_flatbuffer->attributes_as_BatchnormInferenceAttributes();
    ASSERT_NE(packed_attributes, nullptr);

    EXPECT_EQ(packed_attributes->x(), x_tensor->get_uid());
    EXPECT_EQ(packed_attributes->y(), y_tensor->get_uid());
    EXPECT_EQ(packed_attributes->mean(), mean_tensor->get_uid());
    EXPECT_EQ(packed_attributes->inv_variance(), inv_variance_tensor->get_uid());
    EXPECT_EQ(packed_attributes->scale(), scale_tensor->get_uid());
    EXPECT_EQ(packed_attributes->bias(), bias_tensor->get_uid());
}

TEST(BatchnormInferenceNodeTests, PackNodeWithoutMeanAndInvVariance)
{
    Batchnorm_inference_attributes batchnorm_attributes;
    batchnorm_attributes.name = "BatchnormInference";

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_uid(1)
        .set_name("XTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnorm_attributes.set_x(x_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_uid(2)
        .set_name("YTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnorm_attributes.set_y(y_tensor);

    auto scale_tensor = std::make_shared<Tensor_attributes>();
    scale_tensor->set_uid(3)
        .set_name("ScaleTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnorm_attributes.set_scale(scale_tensor);

    auto bias_tensor = std::make_shared<Tensor_attributes>();
    bias_tensor->set_uid(4)
        .set_name("BiasTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnorm_attributes.set_bias(bias_tensor);

    // Do not set mean and inv_variance

    Graph_attributes       graph_attributes;
    BatchnormInferenceNode node(std::move(batchnorm_attributes), graph_attributes);

    flatbuffers::FlatBufferBuilder builder;
    auto                           offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto buffer_pointer  = builder.GetBufferPointer();
    auto node_flatbuffer = flatbuffers::GetRoot<hipdnn::sdk::Node>(buffer_pointer);

    EXPECT_STREQ(node_flatbuffer->name()->c_str(), "BatchnormInference");
    EXPECT_EQ(node_flatbuffer->attributes_type(),
              hipdnn::sdk::NodeAttributes_BatchnormInferenceAttributes);

    auto packed_attributes = node_flatbuffer->attributes_as_BatchnormInferenceAttributes();
    ASSERT_NE(packed_attributes, nullptr);

    EXPECT_EQ(packed_attributes->x(), x_tensor->get_uid());
    EXPECT_EQ(packed_attributes->y(), y_tensor->get_uid());
    EXPECT_EQ(packed_attributes->mean(), flatbuffers::nullopt); // Verify mean is null
    EXPECT_EQ(packed_attributes->inv_variance(),
              flatbuffers::nullopt); // Verify inv_variance is null
    EXPECT_EQ(packed_attributes->scale(), scale_tensor->get_uid());
    EXPECT_EQ(packed_attributes->bias(), bias_tensor->get_uid());
}