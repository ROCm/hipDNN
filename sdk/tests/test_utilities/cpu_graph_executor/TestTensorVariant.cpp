// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <memory>
#include <variant>
#include <vector>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/TensorVariant.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::data_objects;

TEST(TestTensorVariant, CreateHostOnlyShallowTensorTemplateFloat)
{
    std::array<float, 6> backing = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> dims{1, 2, 3};
    std::vector<int64_t> strides{6, 3, 1};
    auto t = createHostOnlyShallowTensor<float>(backing.data(), dims, strides);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->dims(), dims);
    EXPECT_EQ(t->strides(), strides);
    EXPECT_EQ(t->memory().hostData(), backing.data());
}

TEST(TestTensorVariant, CreateFloatShallowTensorVariantInternal)
{
    std::array<float, 4> backing = {0};
    std::vector<int64_t> dims{1, 1, 2, 2};
    std::vector<int64_t> strides{4, 4, 2, 1};
    auto variant
        = createHostOnlyShallowTensorVariantInternal(DataType_FLOAT, backing.data(), dims, strides);
    EXPECT_TRUE(std::holds_alternative<std::unique_ptr<TensorBase<float>>>(variant));
    auto& ptr = *std::get<std::unique_ptr<TensorBase<float>>>(variant);
    EXPECT_EQ(ptr.dims(), dims);
    EXPECT_EQ(ptr.strides(), strides);
    EXPECT_EQ(ptr.memory().hostData(), backing.data());
}

TEST(TestTensorVariant, CreateHalfShallowTensorVariantInternal)
{
    std::array<half, 2> backing = {};
    std::vector<int64_t> dims{1, 1, 1, 2};
    std::vector<int64_t> strides{2, 2, 2, 1};
    auto variant
        = createHostOnlyShallowTensorVariantInternal(DataType_HALF, backing.data(), dims, strides);
    EXPECT_TRUE(std::holds_alternative<std::unique_ptr<TensorBase<half>>>(variant));
    auto& ptr = *std::get<std::unique_ptr<TensorBase<half>>>(variant);
    EXPECT_EQ(ptr.dims(), dims);
    EXPECT_EQ(ptr.strides(), strides);
    EXPECT_EQ(ptr.memory().hostData(), backing.data());
}

TEST(TestTensorVariant, UnsupportedDataTypeThrows)
{
    std::array<int, 1> dummy = {0};
    std::vector<int64_t> dims{1};
    std::vector<int64_t> strides{1};
    EXPECT_THROW(
        createHostOnlyShallowTensorVariantInternal(DataType_INT32, dummy.data(), dims, strides),
        std::runtime_error);
}

TEST(TestTensorVariant, FlatbufferVectorToStdNullReturnsEmpty)
{
    auto v = flatbufferVectorToStd(static_cast<const ::flatbuffers::Vector<int64_t>*>(nullptr));
    EXPECT_TRUE(v.empty());
}

TEST(TestTensorVariant, FlatbufferVectorToStdCopiesValues)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<int64_t> src{5, 4, 3};
    auto fbVec = builder.CreateVector(src);
    builder.Finish(fbVec);
    auto* rootVec = flatbuffers::GetRoot<flatbuffers::Vector<int64_t>>(builder.GetBufferPointer());
    auto out = flatbufferVectorToStd(rootVec);
    EXPECT_EQ(out, src);
}

TEST(TestTensorVariant, CreateHostOnlyShallowTensorVariantFromTensorAttributesFloat)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<int64_t> dims{1, 3, 4, 4};
    std::vector<int64_t> strides{48, 16, 4, 1};
    auto attrOffset
        = CreateTensorAttributesDirect(builder, 7, "x", DataType_FLOAT, &strides, &dims);
    builder.Finish(attrOffset);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    std::array<float, 48> backing = {0};
    auto variant = createHostOnlyShallowTensorVariant(*attr, backing.data());
    ASSERT_TRUE(std::holds_alternative<std::unique_ptr<TensorBase<float>>>(variant));
    auto& t = *std::get<std::unique_ptr<TensorBase<float>>>(variant);
    EXPECT_EQ(t.dims(), dims);
    EXPECT_EQ(t.strides(), strides);
    EXPECT_EQ(t.memory().hostData(), backing.data());
}
