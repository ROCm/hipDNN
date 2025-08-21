// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/batchnorm_backward_attributes.hpp>

TEST(BatchnormBackwardAttributesTests, CreateBatchnormBackwardAttributes)
{
    hipdnn_frontend::graph::Batchnorm_backward_attributes batchnorm_attributes;

    batchnorm_attributes.set_dy(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_x(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_mean(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_inv_variance(
        std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_dx(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_dscale(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_dbias(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());

    auto dy_tensor = batchnorm_attributes.get_dy();
    dy_tensor->set_uid(1)
        .set_name("DyTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto x_tensor = batchnorm_attributes.get_x();
    x_tensor->set_uid(2)
        .set_name("XTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto scale_tensor = batchnorm_attributes.get_scale();
    scale_tensor->set_uid(3)
        .set_name("ScaleTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto mean_tensor = batchnorm_attributes.get_mean();
    mean_tensor->set_uid(4)
        .set_name("MeanTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto inv_variance_tensor = batchnorm_attributes.get_inv_variance();
    inv_variance_tensor->set_uid(5)
        .set_name("InvVarianceTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto dx_tensor = batchnorm_attributes.get_dx();
    dx_tensor->set_uid(6)
        .set_name("DxTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto dscale_tensor = batchnorm_attributes.get_dscale();
    dscale_tensor->set_uid(7)
        .set_name("DscaleTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto dbias_tensor = batchnorm_attributes.get_dbias();
    dbias_tensor->set_uid(8)
        .set_name("DbiasTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto peer_stat_1 = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    peer_stat_1->set_uid(9)
        .set_name("PeerStat1")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2})
        .set_stride({3, 4});

    auto peer_stat_2 = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    peer_stat_2->set_uid(10)
        .set_name("PeerStat2")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({5, 6})
        .set_stride({7, 8});

    batchnorm_attributes.set_peer_stats({peer_stat_1, peer_stat_2});

    EXPECT_EQ(dy_tensor->get_uid(), 1);
    EXPECT_EQ(dy_tensor->get_name(), "DyTensor");
    EXPECT_EQ(dy_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(dy_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(dy_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(x_tensor->get_uid(), 2);
    EXPECT_EQ(x_tensor->get_name(), "XTensor");
    EXPECT_EQ(x_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(x_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(x_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(scale_tensor->get_uid(), 3);
    EXPECT_EQ(scale_tensor->get_name(), "ScaleTensor");
    EXPECT_EQ(scale_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(scale_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(scale_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(mean_tensor->get_uid(), 4);
    EXPECT_EQ(mean_tensor->get_name(), "MeanTensor");
    EXPECT_EQ(mean_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(mean_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(mean_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(inv_variance_tensor->get_uid(), 5);
    EXPECT_EQ(inv_variance_tensor->get_name(), "InvVarianceTensor");
    EXPECT_EQ(inv_variance_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(inv_variance_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(inv_variance_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(dx_tensor->get_uid(), 6);
    EXPECT_EQ(dx_tensor->get_name(), "DxTensor");
    EXPECT_EQ(dx_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(dx_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(dx_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(dscale_tensor->get_uid(), 7);
    EXPECT_EQ(dscale_tensor->get_name(), "DscaleTensor");
    EXPECT_EQ(dscale_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(dscale_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(dscale_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(dbias_tensor->get_uid(), 8);
    EXPECT_EQ(dbias_tensor->get_name(), "DbiasTensor");
    EXPECT_EQ(dbias_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(dbias_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(dbias_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    const auto& peer_stats = batchnorm_attributes.get_peer_stats();
    ASSERT_EQ(peer_stats.size(), 2);

    EXPECT_EQ(peer_stats[0]->get_uid(), 9);
    EXPECT_EQ(peer_stats[0]->get_name(), "PeerStat1");
    EXPECT_EQ(peer_stats[0]->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(peer_stats[0]->get_dim(), (std::vector<int64_t>{1, 2}));
    EXPECT_EQ(peer_stats[0]->get_stride(), (std::vector<int64_t>{3, 4}));

    EXPECT_EQ(peer_stats[1]->get_uid(), 10);
    EXPECT_EQ(peer_stats[1]->get_name(), "PeerStat2");
    EXPECT_EQ(peer_stats[1]->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(peer_stats[1]->get_dim(), (std::vector<int64_t>{5, 6}));
    EXPECT_EQ(peer_stats[1]->get_stride(), (std::vector<int64_t>{7, 8}));
}

TEST(BatchnormBackwardAttributesTests, SetDyWithMove)
{
    hipdnn_frontend::graph::Batchnorm_backward_attributes batchnorm_attributes;

    auto dy_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    dy_tensor->set_uid(1).set_name("DyTensor");

    auto raw_ptr = dy_tensor.get();

    batchnorm_attributes.set_dy(std::move(dy_tensor));

    auto retrieved = batchnorm_attributes.get_dy();
    EXPECT_EQ(retrieved->get_uid(), 1);
    EXPECT_EQ(retrieved->get_name(), "DyTensor");

    EXPECT_EQ(dy_tensor, nullptr);
    EXPECT_EQ(retrieved.get(), raw_ptr);
}

TEST(BatchnormBackwardAttributesTests, SetXWithMove)
{
    hipdnn_frontend::graph::Batchnorm_backward_attributes batchnorm_attributes;

    auto x_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    x_tensor->set_uid(2).set_name("XTensor");

    auto raw_ptr = x_tensor.get();

    batchnorm_attributes.set_x(std::move(x_tensor));

    auto retrieved = batchnorm_attributes.get_x();
    EXPECT_EQ(retrieved->get_uid(), 2);
    EXPECT_EQ(retrieved->get_name(), "XTensor");

    EXPECT_EQ(x_tensor, nullptr);
    EXPECT_EQ(retrieved.get(), raw_ptr);
}

TEST(BatchnormBackwardAttributesTests, SetScaleWithMove)
{
    hipdnn_frontend::graph::Batchnorm_backward_attributes batchnorm_attributes;

    auto scale_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    scale_tensor->set_uid(3).set_name("ScaleTensor");

    auto raw_ptr = scale_tensor.get();

    batchnorm_attributes.set_scale(std::move(scale_tensor));

    auto retrieved = batchnorm_attributes.get_scale();
    EXPECT_EQ(retrieved->get_uid(), 3);
    EXPECT_EQ(retrieved->get_name(), "ScaleTensor");

    EXPECT_EQ(scale_tensor, nullptr);
    EXPECT_EQ(retrieved.get(), raw_ptr);
}

TEST(BatchnormBackwardAttributesTests, SetDxWithMove)
{
    hipdnn_frontend::graph::Batchnorm_backward_attributes batchnorm_attributes;

    auto dx_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    dx_tensor->set_uid(4).set_name("DxTensor");

    auto raw_ptr = dx_tensor.get();

    batchnorm_attributes.set_dx(std::move(dx_tensor));

    auto retrieved = batchnorm_attributes.get_dx();
    EXPECT_EQ(retrieved->get_uid(), 4);
    EXPECT_EQ(retrieved->get_name(), "DxTensor");

    EXPECT_EQ(dx_tensor, nullptr);
    EXPECT_EQ(retrieved.get(), raw_ptr);
}

TEST(BatchnormBackwardAttributesTests, SetPeerStatsWithMove)
{
    hipdnn_frontend::graph::Batchnorm_backward_attributes batchnorm_attributes;

    auto peer_stat_1 = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    peer_stat_1->set_uid(10).set_name("PeerStat1");

    auto peer_stat_2 = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    peer_stat_2->set_uid(11).set_name("PeerStat2");

    auto raw_ptr_1 = peer_stat_1.get();
    auto raw_ptr_2 = peer_stat_2.get();

    std::vector<std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>> peer_stats_move
        = {peer_stat_1, peer_stat_2};

    batchnorm_attributes.set_peer_stats(std::move(peer_stats_move));

    const auto& peer_stats = batchnorm_attributes.get_peer_stats();
    ASSERT_EQ(peer_stats.size(), 2);
    EXPECT_EQ(peer_stats[0]->get_uid(), 10);
    EXPECT_EQ(peer_stats[0]->get_name(), "PeerStat1");
    EXPECT_EQ(peer_stats[1]->get_uid(), 11);
    EXPECT_EQ(peer_stats[1]->get_name(), "PeerStat2");

    // Verify the raw pointers match (same objects were moved)
    EXPECT_EQ(peer_stats[0].get(), raw_ptr_1);
    EXPECT_EQ(peer_stats[1].get(), raw_ptr_2);
}

TEST(BatchnormBackwardAttributesTests, SetSavedMeanAndInvVarianceWithMove)
{
    hipdnn_frontend::graph::Batchnorm_backward_attributes batchnorm_attributes;

    auto mean_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    mean_tensor->set_uid(20).set_name("MeanTensor");

    auto inv_variance_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    inv_variance_tensor->set_uid(21).set_name("InvVarianceTensor");

    auto raw_mean_ptr = mean_tensor.get();
    auto raw_inv_variance_ptr = inv_variance_tensor.get();

    batchnorm_attributes.set_saved_mean_and_inv_variance(std::move(mean_tensor),
                                                         std::move(inv_variance_tensor));

    auto retrieved_mean = batchnorm_attributes.get_mean();
    EXPECT_EQ(retrieved_mean->get_uid(), 20);
    EXPECT_EQ(retrieved_mean->get_name(), "MeanTensor");

    auto retrieved_inv_variance = batchnorm_attributes.get_inv_variance();
    EXPECT_EQ(retrieved_inv_variance->get_uid(), 21);
    EXPECT_EQ(retrieved_inv_variance->get_name(), "InvVarianceTensor");

    // Verify the objects were moved
    EXPECT_EQ(mean_tensor, nullptr);
    EXPECT_EQ(inv_variance_tensor, nullptr);

    // Verify the raw pointers match
    EXPECT_EQ(retrieved_mean.get(), raw_mean_ptr);
    EXPECT_EQ(retrieved_inv_variance.get(), raw_inv_variance_ptr);
}

// Simplified move tests - testing move semantics without setting uid/name

TEST(BatchnormBackwardAttributesTests, SimplifiedSetDyWithMove)
{
    hipdnn_frontend::graph::Batchnorm_backward_attributes batchnorm_attributes;

    auto dy_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    batchnorm_attributes.set_dy(std::move(dy_tensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnorm_attributes.get_dy(), nullptr);
}

TEST(BatchnormBackwardAttributesTests, SimplifiedSetPeerStatsWithMove)
{
    hipdnn_frontend::graph::Batchnorm_backward_attributes batchnorm_attributes;

    std::vector<std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>> peer_stats_move;
    peer_stats_move.push_back(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    peer_stats_move.push_back(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    peer_stats_move.push_back(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());

    size_t original_size = peer_stats_move.size();
    batchnorm_attributes.set_peer_stats(std::move(peer_stats_move));

    // Verify the vector was moved
    const auto& peer_stats = batchnorm_attributes.get_peer_stats();
    EXPECT_EQ(peer_stats.size(), original_size);
}

TEST(BatchnormBackwardAttributesTests, SimplifiedSetSavedMeanAndInvVarianceWithMove)
{
    hipdnn_frontend::graph::Batchnorm_backward_attributes batchnorm_attributes;

    auto mean_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    auto inv_variance_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();

    batchnorm_attributes.set_saved_mean_and_inv_variance(std::move(mean_tensor),
                                                         std::move(inv_variance_tensor));

    // Just verify the tensors were set
    EXPECT_NE(batchnorm_attributes.get_mean(), nullptr);
    EXPECT_NE(batchnorm_attributes.get_inv_variance(), nullptr);
}
