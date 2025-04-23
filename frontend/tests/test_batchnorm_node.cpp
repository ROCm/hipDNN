// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/batchnorm_attributes.hpp>
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_frontend/node/batchnorm_node.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(BatchnormNodeTests, PreValidateNode)
{
    Batchnorm_attributes batchnorm_attributes;
    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_y(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_bias(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_epsilon(std::make_shared<Tensor_attributes>());

    Graph_attributes graph_attributes;
    BatchnormNode    node(std::move(batchnorm_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK);
}

TEST(BatchnormNodeTests, PreValidateNodeMissingValues)
{
    Batchnorm_attributes batchnorm_attributes;

    Graph_attributes graph_attributes;
    BatchnormNode    node(std::move(batchnorm_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    auto          batchnorm_attributes_copy = batchnorm_attributes;
    BatchnormNode node_with_x(std::move(batchnorm_attributes_copy), graph_attributes);

    error = node_with_x.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_y(std::make_shared<Tensor_attributes>());
    batchnorm_attributes_copy = batchnorm_attributes;
    BatchnormNode node_with_y(std::move(batchnorm_attributes_copy), graph_attributes);

    error = node_with_y.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes_copy = batchnorm_attributes;
    BatchnormNode node_with_scale(std::move(batchnorm_attributes_copy), graph_attributes);

    error = node_with_scale.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_bias(std::make_shared<Tensor_attributes>());
    batchnorm_attributes_copy = batchnorm_attributes;
    BatchnormNode node_with_bias(std::move(batchnorm_attributes_copy), graph_attributes);

    error = node_with_bias.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_epsilon(std::make_shared<Tensor_attributes>());
    batchnorm_attributes_copy = batchnorm_attributes;
    BatchnormNode node_with_all_values(std::move(batchnorm_attributes_copy), graph_attributes);

    error = node_with_all_values.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK);
}

TEST(BatchnormNodeTests, InferPropertiesNode)
{
    Batchnorm_attributes batchnorm_attributes;
    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_y(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_bias(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_epsilon(std::make_shared<Tensor_attributes>());

    auto input_tensor = batchnorm_attributes.get_x();
    input_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = batchnorm_attributes.get_y();
    output_tensor->set_uid(2).set_name("OutputTensor");

    Graph_attributes graph_attributes;
    BatchnormNode    node(std::move(batchnorm_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK);

    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(output_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(BatchnormNodeTests, InferPropertiesNodeWithStats)
{
    Batchnorm_attributes batchnorm_attributes;
    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_y(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_bias(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_epsilon(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_mean(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_inv_variance(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_prev_running_mean(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_prev_running_variance(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_next_running_mean(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_next_running_variance(std::make_shared<Tensor_attributes>());

    auto input_tensor = batchnorm_attributes.get_x();
    input_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = batchnorm_attributes.get_y();
    output_tensor->set_uid(2).set_name("OutputTensor");

    auto mean_tensor = batchnorm_attributes.get_mean();
    mean_tensor->set_uid(3).set_name("MeanTensor");

    auto inv_variance_tensor = batchnorm_attributes.get_inv_variance();
    inv_variance_tensor->set_uid(4).set_name("InvVarianceTensor");

    auto next_running_mean_tensor = batchnorm_attributes.get_next_running_mean();
    next_running_mean_tensor->set_uid(5).set_name("NextRunningMeanTensor");

    auto next_running_variance_tensor = batchnorm_attributes.get_next_running_variance();
    next_running_variance_tensor->set_uid(6).set_name("NextRunningVarianceTensor");

    Graph_attributes graph_attributes;
    BatchnormNode    node(std::move(batchnorm_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK);

    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(output_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(mean_tensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(inv_variance_tensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(next_running_mean_tensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(next_running_variance_tensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
}

TEST(BatchnormNodeTests, PackNode)
{
    Batchnorm_attributes batchnorm_attributes;
    batchnorm_attributes.name = "Batchnorm";

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

    auto epsilon_tensor = std::make_shared<Tensor_attributes>();
    epsilon_tensor->set_uid(5)
        .set_name("EpsilonTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1})
        .set_stride({1});
    batchnorm_attributes.set_epsilon(epsilon_tensor);

    Graph_attributes graph_attributes;
    BatchnormNode    node(std::move(batchnorm_attributes), graph_attributes);

    flatbuffers::FlatBufferBuilder builder;
    auto                           offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto buffer_pointer  = builder.GetBufferPointer();
    auto node_flatbuffer = flatbuffers::GetRoot<hipdnn::sdk::Node>(buffer_pointer);

    EXPECT_STREQ(node_flatbuffer->name()->c_str(), "Batchnorm");
    EXPECT_EQ(node_flatbuffer->attributes_type(), hipdnn::sdk::NodeAttributes_BatchnormAttributes);

    auto packed_attributes = node_flatbuffer->attributes_as_BatchnormAttributes();
    ASSERT_NE(packed_attributes, nullptr);

    EXPECT_EQ(packed_attributes->x(), x_tensor->get_uid());
    EXPECT_EQ(packed_attributes->y(), y_tensor->get_uid());
    EXPECT_EQ(packed_attributes->scale(), scale_tensor->get_uid());
    EXPECT_EQ(packed_attributes->bias(), bias_tensor->get_uid());
    EXPECT_EQ(packed_attributes->epsilon(), epsilon_tensor->get_uid());
}

TEST(BatchnormNodeTests, GatherhipdnnTensorIds)
{
    Batchnorm_attributes batchnorm_attributes;
    auto                 x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_uid(1).set_name("XTensor");
    batchnorm_attributes.set_x(x_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_uid(2).set_name("YTensor");
    batchnorm_attributes.set_y(y_tensor);

    auto scale_tensor = std::make_shared<Tensor_attributes>();
    scale_tensor->set_uid(3).set_name("ScaleTensor");
    batchnorm_attributes.set_scale(scale_tensor);

    auto bias_tensor = std::make_shared<Tensor_attributes>();
    bias_tensor->set_uid(4).set_name("BiasTensor");
    batchnorm_attributes.set_bias(bias_tensor);

    auto epsilon_tensor = std::make_shared<Tensor_attributes>();
    epsilon_tensor->set_uid(5).set_name("EpsilonTensor");
    batchnorm_attributes.set_epsilon(epsilon_tensor);

    auto peer_stat_1 = std::make_shared<Tensor_attributes>();
    peer_stat_1->set_uid(9).set_name("PeerStat1");

    auto peer_stat_2 = std::make_shared<Tensor_attributes>();
    peer_stat_2->set_uid(10).set_name("PeerStat2");

    batchnorm_attributes.set_peer_stats({peer_stat_1, peer_stat_2});

    Graph_attributes graph_attributes;
    BatchnormNode    node(std::move(batchnorm_attributes), graph_attributes);

    std::unordered_set<int64_t> used_ids;
    node.gather_hipdnn_tensor_ids(used_ids);

    EXPECT_TRUE(used_ids.find(1) != used_ids.end());
    EXPECT_TRUE(used_ids.find(2) != used_ids.end());
    EXPECT_TRUE(used_ids.find(3) != used_ids.end());
    EXPECT_TRUE(used_ids.find(4) != used_ids.end());
    EXPECT_TRUE(used_ids.find(5) != used_ids.end());
    EXPECT_TRUE(used_ids.find(9) != used_ids.end());
    EXPECT_TRUE(used_ids.find(10) != used_ids.end());
}

TEST(BatchnormNodeTests, PopulatehipdnnTensorIds)
{
    Batchnorm_attributes batchnorm_attributes;
    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_y(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_bias(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_epsilon(std::make_shared<Tensor_attributes>());

    auto peer_stat_1 = std::make_shared<Tensor_attributes>();
    auto peer_stat_2 = std::make_shared<Tensor_attributes>();

    batchnorm_attributes.set_peer_stats({peer_stat_1, peer_stat_2});

    Graph_attributes graph_attributes;
    BatchnormNode    node(std::move(batchnorm_attributes), graph_attributes);

    std::unordered_map<int64_t, std::shared_ptr<Tensor_attributes>> tensor_lookup;
    std::unordered_set<int64_t>                                     used_ids;
    int64_t                                                         current_tensor_id = 1;

    auto error = node.populate_hipdnn_tensor_ids(tensor_lookup, current_tensor_id, used_ids);
    EXPECT_EQ(error.code, error_code_t::OK);

    std::vector<std::shared_ptr<Tensor_attributes>> tensors;
    tensors.reserve(node.attributes.inputs.size() + node.attributes.outputs.size()
                    + node.attributes.peer_stats.size());

    for(const auto& input_pair : node.attributes.inputs)
    {
        tensors.emplace_back(input_pair.second);
    }

    for(const auto& output_pair : node.attributes.outputs)
    {
        tensors.emplace_back(output_pair.second);
    }

    for(const auto& peer_stat : node.attributes.peer_stats)
    {
        tensors.emplace_back(peer_stat);
    }

    std::unordered_set<int64_t> tensor_ids;
    for(const auto& tensor : tensors)
    {
        ASSERT_TRUE(tensor->has_uid());
        EXPECT_TRUE(tensor_ids.insert(tensor->get_uid()).second)
            << "Duplicate tensor ID found: " << tensor->get_uid();
    }
}