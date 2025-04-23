// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/batchnorm_attributes.hpp>

TEST(BatchnormAttributesTests, CreateBatchnormAttributes)
{
    hipdnn_frontend::graph::Batchnorm_attributes batchnorm_attributes;

    batchnorm_attributes.set_x(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_y(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_scale(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_bias(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_prev_running_mean(
        std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_prev_running_variance(
        std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_momentum(
        std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_mean(std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_inv_variance(
        std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_next_running_mean(
        std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_next_running_variance(
        std::make_shared<hipdnn_frontend::graph::Tensor_attributes>());
    batchnorm_attributes.set_epsilon(
        std::make_shared<hipdnn_frontend::graph::Tensor_attributes>()); // Set epsilon

    auto x_tensor = batchnorm_attributes.get_x();
    x_tensor->set_uid(1)
        .set_name("XTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto y_tensor = batchnorm_attributes.get_y();
    y_tensor->set_uid(2)
        .set_name("YTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto scale_tensor = batchnorm_attributes.get_scale();
    scale_tensor->set_uid(3)
        .set_name("ScaleTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto bias_tensor = batchnorm_attributes.get_bias();
    bias_tensor->set_uid(4)
        .set_name("BiasTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto prev_mean_tensor = batchnorm_attributes.get_prev_running_mean();
    prev_mean_tensor->set_uid(5)
        .set_name("PrevMeanTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto prev_variance_tensor = batchnorm_attributes.get_prev_running_variance();
    prev_variance_tensor->set_uid(6)
        .set_name("PrevVarianceTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto momentum_tensor = batchnorm_attributes.get_momentum();
    momentum_tensor->set_uid(7)
        .set_name("MomentumTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto mean_tensor = batchnorm_attributes.get_mean();
    mean_tensor->set_uid(8)
        .set_name("MeanTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto inv_variance_tensor = batchnorm_attributes.get_inv_variance();
    inv_variance_tensor->set_uid(9)
        .set_name("InvVarianceTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto next_mean_tensor = batchnorm_attributes.get_next_running_mean();
    next_mean_tensor->set_uid(10)
        .set_name("NextMeanTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto next_variance_tensor = batchnorm_attributes.get_next_running_variance();
    next_variance_tensor->set_uid(11)
        .set_name("NextVarianceTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto epsilon_tensor = batchnorm_attributes.get_epsilon();
    epsilon_tensor->set_uid(14)
        .set_name("EpsilonTensor")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1})
        .set_stride({1});

    auto peer_stat_1 = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    peer_stat_1->set_uid(12)
        .set_name("PeerStat1")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({1, 2})
        .set_stride({3, 4});

    auto peer_stat_2 = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    peer_stat_2->set_uid(13)
        .set_name("PeerStat2")
        .set_data_type(hipdnn_frontend::DataType_t::FLOAT)
        .set_dim({5, 6})
        .set_stride({7, 8});

    batchnorm_attributes.set_peer_stats({peer_stat_1, peer_stat_2});

    const auto& peer_stats = batchnorm_attributes.get_peer_stats();
    ASSERT_EQ(peer_stats.size(), 2);

    EXPECT_EQ(peer_stats[0]->get_uid(), 12);
    EXPECT_EQ(peer_stats[0]->get_name(), "PeerStat1");
    EXPECT_EQ(peer_stats[0]->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(peer_stats[0]->get_dim(), (std::vector<int64_t>{1, 2}));
    EXPECT_EQ(peer_stats[0]->get_stride(), (std::vector<int64_t>{3, 4}));

    EXPECT_EQ(peer_stats[1]->get_uid(), 13);
    EXPECT_EQ(peer_stats[1]->get_name(), "PeerStat2");
    EXPECT_EQ(peer_stats[1]->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(peer_stats[1]->get_dim(), (std::vector<int64_t>{5, 6}));
    EXPECT_EQ(peer_stats[1]->get_stride(), (std::vector<int64_t>{7, 8}));

    EXPECT_EQ(x_tensor->get_uid(), 1);
    EXPECT_EQ(x_tensor->get_name(), "XTensor");
    EXPECT_EQ(x_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(x_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(x_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(y_tensor->get_uid(), 2);
    EXPECT_EQ(y_tensor->get_name(), "YTensor");
    EXPECT_EQ(y_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(y_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(y_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(scale_tensor->get_uid(), 3);
    EXPECT_EQ(scale_tensor->get_name(), "ScaleTensor");
    EXPECT_EQ(scale_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(scale_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(scale_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(bias_tensor->get_uid(), 4);
    EXPECT_EQ(bias_tensor->get_name(), "BiasTensor");
    EXPECT_EQ(bias_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(bias_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(bias_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(prev_mean_tensor->get_uid(), 5);
    EXPECT_EQ(prev_mean_tensor->get_name(), "PrevMeanTensor");
    EXPECT_EQ(prev_mean_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(prev_mean_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(prev_mean_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(prev_variance_tensor->get_uid(), 6);
    EXPECT_EQ(prev_variance_tensor->get_name(), "PrevVarianceTensor");
    EXPECT_EQ(prev_variance_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(prev_variance_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(prev_variance_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(momentum_tensor->get_uid(), 7);
    EXPECT_EQ(momentum_tensor->get_name(), "MomentumTensor");
    EXPECT_EQ(momentum_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(momentum_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(momentum_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(mean_tensor->get_uid(), 8);
    EXPECT_EQ(mean_tensor->get_name(), "MeanTensor");
    EXPECT_EQ(mean_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(mean_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(mean_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(inv_variance_tensor->get_uid(), 9);
    EXPECT_EQ(inv_variance_tensor->get_name(), "InvVarianceTensor");
    EXPECT_EQ(inv_variance_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(inv_variance_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(inv_variance_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(next_mean_tensor->get_uid(), 10);
    EXPECT_EQ(next_mean_tensor->get_name(), "NextMeanTensor");
    EXPECT_EQ(next_mean_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(next_mean_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(next_mean_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(next_variance_tensor->get_uid(), 11);
    EXPECT_EQ(next_variance_tensor->get_name(), "NextVarianceTensor");
    EXPECT_EQ(next_variance_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(next_variance_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(next_variance_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(epsilon_tensor->get_uid(), 14);
    EXPECT_EQ(epsilon_tensor->get_name(), "EpsilonTensor");
    EXPECT_EQ(epsilon_tensor->get_data_type(), hipdnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(epsilon_tensor->get_dim(), (std::vector<int64_t>{1}));
    EXPECT_EQ(epsilon_tensor->get_stride(), (std::vector<int64_t>{1}));
}

TEST(BatchnormAttributesTests, PackAttributes)
{
    hipdnn_frontend::graph::Batchnorm_attributes batchnorm_attributes;

    auto x_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    x_tensor->set_uid(1);
    batchnorm_attributes.set_x(x_tensor);

    auto y_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    y_tensor->set_uid(2);
    batchnorm_attributes.set_y(y_tensor);

    auto scale_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    scale_tensor->set_uid(3);
    batchnorm_attributes.set_scale(scale_tensor);

    auto bias_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    bias_tensor->set_uid(4);
    batchnorm_attributes.set_bias(bias_tensor);

    auto prev_mean_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    prev_mean_tensor->set_uid(5);
    batchnorm_attributes.set_prev_running_mean(prev_mean_tensor);

    auto prev_variance_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    prev_variance_tensor->set_uid(6);
    batchnorm_attributes.set_prev_running_variance(prev_variance_tensor);

    auto momentum_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    momentum_tensor->set_uid(7);
    batchnorm_attributes.set_momentum(momentum_tensor);

    auto mean_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    mean_tensor->set_uid(8);
    batchnorm_attributes.set_mean(mean_tensor);

    auto inv_variance_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    inv_variance_tensor->set_uid(9);
    batchnorm_attributes.set_inv_variance(inv_variance_tensor);

    auto next_mean_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    next_mean_tensor->set_uid(10);
    batchnorm_attributes.set_next_running_mean(next_mean_tensor);

    auto next_variance_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    next_variance_tensor->set_uid(11);
    batchnorm_attributes.set_next_running_variance(next_variance_tensor);

    auto epsilon_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    epsilon_tensor->set_uid(14);
    batchnorm_attributes.set_epsilon(epsilon_tensor);

    auto peer_stat_1 = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    peer_stat_1->set_uid(12);

    auto peer_stat_2 = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    peer_stat_2->set_uid(13);

    batchnorm_attributes.set_peer_stats({peer_stat_1, peer_stat_2});

    flatbuffers::FlatBufferBuilder builder;
    auto packed_attributes = batchnorm_attributes.pack_attributes(builder);
    builder.Finish(packed_attributes);

    auto buffer = builder.GetBufferPointer();
    auto batchnorm_attributes_fb
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::BatchnormAttributes>(buffer);

    EXPECT_EQ(batchnorm_attributes_fb->x(), 1);
    EXPECT_EQ(batchnorm_attributes_fb->y(), 2);
    EXPECT_EQ(batchnorm_attributes_fb->scale(), 3);
    EXPECT_EQ(batchnorm_attributes_fb->bias(), 4);
    EXPECT_EQ(batchnorm_attributes_fb->prev_running_mean(), 5);
    EXPECT_EQ(batchnorm_attributes_fb->prev_running_variance(), 6);
    EXPECT_EQ(batchnorm_attributes_fb->momentum(), 7);
    EXPECT_EQ(batchnorm_attributes_fb->mean(), 8);
    EXPECT_EQ(batchnorm_attributes_fb->inv_variance(), 9);
    EXPECT_EQ(batchnorm_attributes_fb->next_running_mean(), 10);
    EXPECT_EQ(batchnorm_attributes_fb->next_running_variance(), 11);
    EXPECT_EQ(batchnorm_attributes_fb->epsilon(), 14);

    ASSERT_EQ(batchnorm_attributes_fb->peer_stats()->size(), 2);
    EXPECT_EQ(batchnorm_attributes_fb->peer_stats()->Get(0), 12);
    EXPECT_EQ(batchnorm_attributes_fb->peer_stats()->Get(1), 13);
}

TEST(BatchnormAttributesTests, PackAttributesWithoutOptionalValues)
{
    hipdnn_frontend::graph::Batchnorm_attributes batchnorm_attributes;

    auto x_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    x_tensor->set_uid(1);
    batchnorm_attributes.set_x(x_tensor);

    auto y_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    y_tensor->set_uid(2);
    batchnorm_attributes.set_y(y_tensor);

    auto scale_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    scale_tensor->set_uid(3);
    batchnorm_attributes.set_scale(scale_tensor);

    auto bias_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    bias_tensor->set_uid(4);
    batchnorm_attributes.set_bias(bias_tensor);

    auto epsilon_tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    epsilon_tensor->set_uid(5);
    batchnorm_attributes.set_epsilon(epsilon_tensor);

    flatbuffers::FlatBufferBuilder builder;
    auto packed_attributes = batchnorm_attributes.pack_attributes(builder);
    builder.Finish(packed_attributes);

    auto buffer = builder.GetBufferPointer();
    auto batchnorm_attributes_fb
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::BatchnormAttributes>(buffer);

    EXPECT_EQ(batchnorm_attributes_fb->x(), 1);
    EXPECT_EQ(batchnorm_attributes_fb->y(), 2);
    EXPECT_EQ(batchnorm_attributes_fb->scale(), 3);
    EXPECT_EQ(batchnorm_attributes_fb->bias(), 4);
    EXPECT_EQ(batchnorm_attributes_fb->epsilon(), 5);

    EXPECT_EQ(batchnorm_attributes_fb->prev_running_mean(), flatbuffers::nullopt);
    EXPECT_EQ(batchnorm_attributes_fb->prev_running_variance(), flatbuffers::nullopt);
    EXPECT_EQ(batchnorm_attributes_fb->momentum(), flatbuffers::nullopt);
    EXPECT_EQ(batchnorm_attributes_fb->mean(), flatbuffers::nullopt);
    EXPECT_EQ(batchnorm_attributes_fb->inv_variance(), flatbuffers::nullopt);
    EXPECT_EQ(batchnorm_attributes_fb->next_running_mean(), flatbuffers::nullopt);
    EXPECT_EQ(batchnorm_attributes_fb->next_running_variance(), flatbuffers::nullopt);

    ASSERT_EQ(batchnorm_attributes_fb->peer_stats()->size(), 0);
}
