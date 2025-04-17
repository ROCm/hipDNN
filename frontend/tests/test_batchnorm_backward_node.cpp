// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/batchnorm_backward_attributes.hpp>
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_frontend/node/batchnorm_backward_node.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(DBNNodeTests, PreValidateNode)
{
    Batchnorm_backward_attributes batchnorm_attributes;
    batchnorm_attributes.set_dy(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_dx(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_dscale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_dbias(std::make_shared<Tensor_attributes>());

    Graph_attributes graph_attributes;
    DBNNode          node(std::move(batchnorm_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK);
}

TEST(DBNNodeTests, PreValidateNodeMissingValues)
{
    Batchnorm_backward_attributes batchnorm_attributes;

    Graph_attributes graph_attributes;
    DBNNode          node(std::move(batchnorm_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_dy(std::make_shared<Tensor_attributes>());
    auto    batchnorm_attributes_copy = batchnorm_attributes;
    DBNNode node_with_dy(std::move(batchnorm_attributes_copy), graph_attributes);

    error = node_with_dy.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    batchnorm_attributes_copy = batchnorm_attributes;
    DBNNode node_with_x(std::move(batchnorm_attributes_copy), graph_attributes);

    error = node_with_x.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes_copy = batchnorm_attributes;
    DBNNode node_with_scale(std::move(batchnorm_attributes_copy), graph_attributes);

    error = node_with_scale.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_dx(std::make_shared<Tensor_attributes>());
    batchnorm_attributes_copy = batchnorm_attributes;
    DBNNode node_with_dx(std::move(batchnorm_attributes_copy), graph_attributes);

    error = node_with_dx.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_dscale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes_copy = batchnorm_attributes;
    DBNNode node_with_dscale(std::move(batchnorm_attributes_copy), graph_attributes);

    error = node_with_dscale.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    batchnorm_attributes.set_dbias(std::make_shared<Tensor_attributes>());
    batchnorm_attributes_copy = batchnorm_attributes;
    DBNNode node_with_all_values(std::move(batchnorm_attributes_copy), graph_attributes);

    error = node_with_all_values.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK);
}

TEST(DBNNodeTests, InferPropertiesNode)
{
    Batchnorm_backward_attributes batchnorm_attributes;
    batchnorm_attributes.set_dy(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_mean(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_inv_variance(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_dx(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_dscale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_dbias(std::make_shared<Tensor_attributes>());

    auto x_tensor = batchnorm_attributes.get_x();
    x_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto dx_tensor = batchnorm_attributes.get_dx();
    dx_tensor->set_uid(2).set_name("DxTensor");

    auto dscale_tensor = batchnorm_attributes.get_dscale();
    dscale_tensor->set_uid(3).set_name("DscaleTensor");

    auto dbias_tensor = batchnorm_attributes.get_dbias();
    dbias_tensor->set_uid(4).set_name("DbiasTensor");

    Graph_attributes graph_attributes;
    DBNNode          node(std::move(batchnorm_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK);

    EXPECT_EQ(dx_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(dx_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(dscale_tensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(dscale_tensor->get_stride(), (std::vector<int64_t>{2, 1, 2, 2}));

    EXPECT_EQ(dbias_tensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(dbias_tensor->get_stride(), (std::vector<int64_t>{2, 1, 2, 2}));
}

TEST(DBNNodeTests, GatherHipdnnTensorIds)
{
    Batchnorm_backward_attributes batchnorm_attributes;
    batchnorm_attributes.set_dy(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_mean(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_inv_variance(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_dx(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_dscale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_dbias(std::make_shared<Tensor_attributes>());

    auto peer_stat_1 = std::make_shared<Tensor_attributes>();
    peer_stat_1->set_uid(9).set_name("PeerStat1");

    auto peer_stat_2 = std::make_shared<Tensor_attributes>();
    peer_stat_2->set_uid(10).set_name("PeerStat2");

    batchnorm_attributes.set_peer_stats({peer_stat_1, peer_stat_2});

    Graph_attributes graph_attributes;
    DBNNode          node(std::move(batchnorm_attributes), graph_attributes);

    std::unordered_set<int64_t> used_ids;
    node.gather_hipdnn_tensor_ids(used_ids);

    EXPECT_TRUE(used_ids.find(9) != used_ids.end());
    EXPECT_TRUE(used_ids.find(10) != used_ids.end());
}

TEST(DBNNodeTests, PopulateHipdnnTensorIds)
{
    Batchnorm_backward_attributes batchnorm_attributes;
    batchnorm_attributes.set_dy(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_x(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_mean(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_inv_variance(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_dx(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_dscale(std::make_shared<Tensor_attributes>());
    batchnorm_attributes.set_dbias(std::make_shared<Tensor_attributes>());

    auto peer_stat_1 = std::make_shared<Tensor_attributes>();
    auto peer_stat_2 = std::make_shared<Tensor_attributes>();

    batchnorm_attributes.set_peer_stats({peer_stat_1, peer_stat_2});

    Graph_attributes graph_attributes;
    DBNNode          node(std::move(batchnorm_attributes), graph_attributes);

    std::unordered_map<int64_t, std::shared_ptr<Tensor_attributes>> tensor_lookup;
    std::unordered_set<int64_t>                                     used_ids;
    int64_t                                                         current_tensor_id = 1;

    auto error = node.populate_hipdnn_tensor_ids(tensor_lookup, current_tensor_id, used_ids);
    EXPECT_EQ(error.code, error_code_t::OK);

    // Collect all tensor attributes from input map, output map, and peer_stats vector
    std::vector<std::shared_ptr<Tensor_attributes>> tensors;
    tensors.reserve(node.attributes.inputs.size() + node.attributes.outputs.size()
                    + node.attributes.peer_stats.size());

    // Add tensors from input map
    for(const auto& input_pair : node.attributes.inputs)
    {
        tensors.emplace_back(input_pair.second);
    }

    // Add tensors from output map
    for(const auto& output_pair : node.attributes.outputs)
    {
        tensors.emplace_back(output_pair.second);
    }

    // Add tensors from peer_stats vector
    for(const auto& peer_stat : node.attributes.peer_stats)
    {
        tensors.emplace_back(peer_stat);
    }

    // Check that all tensors have unique IDs
    std::unordered_set<int64_t> tensor_ids;
    for(const auto& tensor : tensors)
    {
        ASSERT_TRUE(tensor->has_uid());
        EXPECT_TRUE(tensor_ids.insert(tensor->get_uid()).second)
            << "Duplicate tensor ID found: " << tensor->get_uid();
    }
}

TEST(DBNNodeTests, PackNode)
{
    Batchnorm_backward_attributes batchnorm_attributes;
    batchnorm_attributes.name = "BatchnormBackward";

    // Set up tensor attributes
    auto dy_tensor = std::make_shared<Tensor_attributes>();
    dy_tensor->set_uid(1)
        .set_name("DyTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnorm_attributes.set_dy(dy_tensor);

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_uid(2)
        .set_name("XTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnorm_attributes.set_x(x_tensor);

    auto scale_tensor = std::make_shared<Tensor_attributes>();
    scale_tensor->set_uid(3)
        .set_name("ScaleTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnorm_attributes.set_scale(scale_tensor);

    auto mean_tensor = std::make_shared<Tensor_attributes>();
    mean_tensor->set_uid(4)
        .set_name("MeanTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnorm_attributes.set_mean(mean_tensor);

    auto inv_variance_tensor = std::make_shared<Tensor_attributes>();
    inv_variance_tensor->set_uid(5)
        .set_name("InvVarianceTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnorm_attributes.set_inv_variance(inv_variance_tensor);

    auto dx_tensor = std::make_shared<Tensor_attributes>();
    dx_tensor->set_uid(6)
        .set_name("DxTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnorm_attributes.set_dx(dx_tensor);

    auto dscale_tensor = std::make_shared<Tensor_attributes>();
    dscale_tensor->set_uid(7)
        .set_name("DscaleTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnorm_attributes.set_dscale(dscale_tensor);

    auto dbias_tensor = std::make_shared<Tensor_attributes>();
    dbias_tensor->set_uid(8)
        .set_name("DbiasTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnorm_attributes.set_dbias(dbias_tensor);

    Graph_attributes graph_attributes;
    DBNNode          node(std::move(batchnorm_attributes), graph_attributes);

    // Pack the node
    flatbuffers::FlatBufferBuilder builder;
    auto                           offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto buffer_pointer  = builder.GetBufferPointer();
    auto node_flatbuffer = flatbuffers::GetRoot<hipdnn::sdk::Node>(buffer_pointer);

    EXPECT_STREQ(node_flatbuffer->name()->c_str(), "BatchnormBackward");
    EXPECT_EQ(node_flatbuffer->attributes_type(),
              hipdnn::sdk::NodeAttributes_BatchnormBackwardAttributes);

    auto packed_attributes = node_flatbuffer->attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(packed_attributes, nullptr);

    EXPECT_EQ(packed_attributes->dy(), dy_tensor->get_uid());
    EXPECT_EQ(packed_attributes->x(), x_tensor->get_uid());
    EXPECT_EQ(packed_attributes->scale(), scale_tensor->get_uid());
    EXPECT_EQ(packed_attributes->mean(), mean_tensor->get_uid());
    EXPECT_EQ(packed_attributes->inv_variance(), inv_variance_tensor->get_uid());
    EXPECT_EQ(packed_attributes->dx(), dx_tensor->get_uid());
    EXPECT_EQ(packed_attributes->dscale(), dscale_tensor->get_uid());
    EXPECT_EQ(packed_attributes->dbias(), dbias_tensor->get_uid());
}

TEST(DBNNodeTests, PackNodeWithoutMeanAndInvVariance)
{
    Batchnorm_backward_attributes batchnorm_attributes;
    batchnorm_attributes.name = "BatchnormBackward";

    // Set up tensor attributes
    auto dy_tensor = std::make_shared<Tensor_attributes>();
    dy_tensor->set_uid(1)
        .set_name("DyTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnorm_attributes.set_dy(dy_tensor);

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_uid(2)
        .set_name("XTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnorm_attributes.set_x(x_tensor);

    auto scale_tensor = std::make_shared<Tensor_attributes>();
    scale_tensor->set_uid(3)
        .set_name("ScaleTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnorm_attributes.set_scale(scale_tensor);

    auto dx_tensor = std::make_shared<Tensor_attributes>();
    dx_tensor->set_uid(4)
        .set_name("DxTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnorm_attributes.set_dx(dx_tensor);

    auto dscale_tensor = std::make_shared<Tensor_attributes>();
    dscale_tensor->set_uid(5)
        .set_name("DscaleTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnorm_attributes.set_dscale(dscale_tensor);

    auto dbias_tensor = std::make_shared<Tensor_attributes>();
    dbias_tensor->set_uid(6)
        .set_name("DbiasTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnorm_attributes.set_dbias(dbias_tensor);

    Graph_attributes graph_attributes;
    DBNNode          node(std::move(batchnorm_attributes), graph_attributes);

    flatbuffers::FlatBufferBuilder builder;
    auto                           offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto buffer_pointer  = builder.GetBufferPointer();
    auto node_flatbuffer = flatbuffers::GetRoot<hipdnn::sdk::Node>(buffer_pointer);

    EXPECT_STREQ(node_flatbuffer->name()->c_str(), "BatchnormBackward");
    EXPECT_EQ(node_flatbuffer->attributes_type(),
              hipdnn::sdk::NodeAttributes_BatchnormBackwardAttributes);

    auto packed_attributes = node_flatbuffer->attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(packed_attributes, nullptr);

    EXPECT_EQ(packed_attributes->dy(), dy_tensor->get_uid());
    EXPECT_EQ(packed_attributes->x(), x_tensor->get_uid());
    EXPECT_EQ(packed_attributes->scale(), scale_tensor->get_uid());
    EXPECT_EQ(packed_attributes->mean(), flatbuffers::nullopt);
    EXPECT_EQ(packed_attributes->inv_variance(), flatbuffers::nullopt);
    EXPECT_EQ(packed_attributes->dx(), dx_tensor->get_uid());
    EXPECT_EQ(packed_attributes->dscale(), dscale_tensor->get_uid());
    EXPECT_EQ(packed_attributes->dbias(), dbias_tensor->get_uid());
}
