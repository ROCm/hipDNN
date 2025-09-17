// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <type_traits>
#include <unordered_set>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignature.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;

namespace
{
template <class Sig>
constexpr BatchnormSignatureKey makeKey()
{
    return {.inputDataType = Sig::INPUT_DATA_TYPE,
            .scaleBiasDataType = Sig::SCALE_BIAS_DATA_TYPE,
            .meanVarianceDataType = Sig::MEAN_VARIANCE_DATA_TYPE,
            .nodeAttributesType = Sig::NODE_ATTRIBUTES_TYPE};
}
}

TEST(TestBatchnormSignature, StaticValuesFloatSignature)
{
    static_assert(FwdBatchnormSignatureFloat::INPUT_DATA_TYPE == DataType::FLOAT);
    static_assert(FwdBatchnormSignatureFloat::SCALE_BIAS_DATA_TYPE == DataType::FLOAT);
    static_assert(FwdBatchnormSignatureFloat::MEAN_VARIANCE_DATA_TYPE == DataType::FLOAT);
    static_assert(FwdBatchnormSignatureFloat::NODE_ATTRIBUTES_TYPE
                  == NodeAttributes::BatchnormInferenceAttributes);

    auto key = makeKey<FwdBatchnormSignatureFloat>();

    EXPECT_EQ(key.inputDataType, DataType::FLOAT);
    EXPECT_EQ(key.scaleBiasDataType, DataType::FLOAT);
    EXPECT_EQ(key.meanVarianceDataType, DataType::FLOAT);
    EXPECT_EQ(key.nodeAttributesType, NodeAttributes::BatchnormInferenceAttributes);
}

TEST(TestBatchnormSignature, StaticValuesHalfSignature)
{
    static_assert(FwdBatchnormSignatureHalf::INPUT_DATA_TYPE == DataType::HALF);
    static_assert(FwdBatchnormSignatureHalf::SCALE_BIAS_DATA_TYPE == DataType::HALF);
    static_assert(FwdBatchnormSignatureHalf::MEAN_VARIANCE_DATA_TYPE == DataType::HALF);
    static_assert(FwdBatchnormSignatureHalf::NODE_ATTRIBUTES_TYPE
                  == NodeAttributes::BatchnormInferenceAttributes);

    auto key = makeKey<FwdBatchnormSignatureHalf>();

    EXPECT_EQ(key.inputDataType, DataType::HALF);
    EXPECT_EQ(key.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(key.meanVarianceDataType, DataType::HALF);
    EXPECT_EQ(key.nodeAttributesType, NodeAttributes::BatchnormInferenceAttributes);
}

TEST(TestBatchnormSignature, KeyEqualityAndInequality)
{
    BatchnormSignatureKey a{.inputDataType = DataType::FLOAT,
                            .scaleBiasDataType = DataType::FLOAT,
                            .meanVarianceDataType = DataType::FLOAT,
                            .nodeAttributesType = NodeAttributes::BatchnormInferenceAttributes};
    BatchnormSignatureKey b = a;
    BatchnormSignatureKey c{.inputDataType = DataType::HALF,
                            .scaleBiasDataType = DataType::HALF,
                            .meanVarianceDataType = DataType::HALF,
                            .nodeAttributesType = NodeAttributes::BatchnormInferenceAttributes};

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST(TestBatchnormSignature, HashConsistency)
{
    BatchnormSignatureKey a{.inputDataType = DataType::FLOAT,
                            .scaleBiasDataType = DataType::FLOAT,
                            .meanVarianceDataType = DataType::FLOAT,
                            .nodeAttributesType = NodeAttributes::BatchnormInferenceAttributes};
    BatchnormSignatureKey b = a;
    BatchnormSignatureKey c{.inputDataType = DataType::HALF,
                            .scaleBiasDataType = DataType::HALF,
                            .meanVarianceDataType = DataType::HALF,
                            .nodeAttributesType = NodeAttributes::BatchnormInferenceAttributes};

    std::hash<BatchnormSignatureKey> h;
    auto ha = h(a);
    auto hb = h(b);
    auto hc = h(c);

    EXPECT_EQ(ha, hb);
    EXPECT_NE(ha, hc);
}

TEST(TestBatchnormSignature, UnorderedSetBehavior)
{
    BatchnormSignatureKey a{.inputDataType = DataType::FLOAT,
                            .scaleBiasDataType = DataType::FLOAT,
                            .meanVarianceDataType = DataType::FLOAT,
                            .nodeAttributesType = NodeAttributes::BatchnormInferenceAttributes};
    BatchnormSignatureKey b = a;
    BatchnormSignatureKey c{.inputDataType = DataType::HALF,
                            .scaleBiasDataType = DataType::HALF,
                            .meanVarianceDataType = DataType::HALF,
                            .nodeAttributesType = NodeAttributes::BatchnormInferenceAttributes};

    std::unordered_set<BatchnormSignatureKey> set;
    set.insert(a);
    set.insert(b); // duplicate
    set.insert(c);
    EXPECT_EQ(set.size(), 2u);
    EXPECT_TRUE(set.contains(a));
    EXPECT_TRUE(set.contains(c));
}

TEST(TestBatchnormSignature, VariantToKeyVisit)
{
    BatchnormSignatureVariants vFloat = FwdBatchnormSignatureFloat{};
    BatchnormSignatureVariants vHalf = FwdBatchnormSignatureHalf{};

    auto toKey = [](const auto& sig) {
        using Sig = std::decay_t<decltype(sig)>;
        return makeKey<Sig>();
    };

    auto kFloat = std::visit(toKey, vFloat);
    auto kHalf = std::visit(toKey, vHalf);

    EXPECT_EQ(kFloat.inputDataType, DataType::FLOAT);
    EXPECT_EQ(kHalf.inputDataType, DataType::HALF);
    EXPECT_NE(kFloat.inputDataType, kHalf.inputDataType);
    EXPECT_EQ(kFloat.nodeAttributesType, NodeAttributes::BatchnormInferenceAttributes);
    EXPECT_EQ(kHalf.nodeAttributesType, NodeAttributes::BatchnormInferenceAttributes);
}
