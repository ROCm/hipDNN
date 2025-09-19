// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignatureKey.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;

TEST(TestBatchnormSignatureKey, EqualityOperator)
{
    BatchnormSignatureKey key1{.inputDataType = DataType::FLOAT,
                               .scaleBiasDataType = DataType::FLOAT,
                               .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureKey key2{.inputDataType = DataType::FLOAT,
                               .scaleBiasDataType = DataType::FLOAT,
                               .meanVarianceDataType = DataType::FLOAT};
    EXPECT_EQ(key1, key2);

    BatchnormSignatureKey key3{.inputDataType = DataType::HALF,
                               .scaleBiasDataType = DataType::FLOAT,
                               .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureKey key4{.inputDataType = DataType::HALF,
                               .scaleBiasDataType = DataType::FLOAT,
                               .meanVarianceDataType = DataType::FLOAT};
    EXPECT_EQ(key3, key4);

    BatchnormSignatureKey key5{.inputDataType = DataType::FLOAT,
                               .scaleBiasDataType = DataType::FLOAT,
                               .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureKey key6{.inputDataType = DataType::HALF,
                               .scaleBiasDataType = DataType::FLOAT,
                               .meanVarianceDataType = DataType::FLOAT};
    EXPECT_NE(key5, key6);

    BatchnormSignatureKey key7{.inputDataType = DataType::FLOAT,
                               .scaleBiasDataType = DataType::FLOAT,
                               .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureKey key8{.inputDataType = DataType::FLOAT,
                               .scaleBiasDataType = DataType::HALF,
                               .meanVarianceDataType = DataType::FLOAT};
    EXPECT_NE(key7, key8);

    BatchnormSignatureKey key9{.inputDataType = DataType::FLOAT,
                               .scaleBiasDataType = DataType::FLOAT,
                               .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureKey key10{.inputDataType = DataType::FLOAT,
                                .scaleBiasDataType = DataType::FLOAT,
                                .meanVarianceDataType = DataType::DOUBLE};
    EXPECT_NE(key9, key10);
}

TEST(TestBatchnormSignatureKey, HashFunction)
{
    BatchnormSignatureKey key1{.inputDataType = DataType::FLOAT,
                               .scaleBiasDataType = DataType::FLOAT,
                               .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureKey key2{.inputDataType = DataType::FLOAT,
                               .scaleBiasDataType = DataType::FLOAT,
                               .meanVarianceDataType = DataType::FLOAT};

    std::hash<BatchnormSignatureKey> hasher;
    EXPECT_EQ(hasher(key1), hasher(key2));

    BatchnormSignatureKey key3{.inputDataType = DataType::HALF,
                               .scaleBiasDataType = DataType::FLOAT,
                               .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureKey key4{.inputDataType = DataType::FLOAT,
                               .scaleBiasDataType = DataType::HALF,
                               .meanVarianceDataType = DataType::FLOAT};
    BatchnormSignatureKey key5{.inputDataType = DataType::FLOAT,
                               .scaleBiasDataType = DataType::FLOAT,
                               .meanVarianceDataType = DataType::HALF};

    auto hash3 = hasher(key3);
    auto hash4 = hasher(key4);
    auto hash5 = hasher(key5);

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestBatchnormSignatureKey, CopyAndAssignment)
{
    // Test copy constructor
    BatchnormSignatureKey original{.inputDataType = DataType::FLOAT,
                                   .scaleBiasDataType = DataType::HALF,
                                   .meanVarianceDataType = DataType::DOUBLE};
    BatchnormSignatureKey copied(original);

    EXPECT_EQ(original, copied);
    EXPECT_EQ(copied.inputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(copied.meanVarianceDataType, DataType::DOUBLE);

    // Test assignment operator
    BatchnormSignatureKey assigned{.inputDataType = DataType::INT32,
                                   .scaleBiasDataType = DataType::INT32,
                                   .meanVarianceDataType = DataType::INT32};
    assigned = original;

    EXPECT_EQ(assigned, original);
    EXPECT_EQ(assigned.inputDataType, DataType::FLOAT);
    EXPECT_EQ(assigned.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(assigned.meanVarianceDataType, DataType::DOUBLE);
}
