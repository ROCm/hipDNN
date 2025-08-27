// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "FlatbufferTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TestMacros.hpp"
#include "descriptors/DescriptorFactory.hpp"
#include "descriptors/EngineConfigDescriptor.hpp"
#include "descriptors/EngineDescriptor.hpp"
#include "descriptors/ExecutionPlanDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/VariantDescriptor.hpp"
#include "handle/HandleFactory.hpp"

#include "gtest/gtest.h"

using namespace hipdnn_backend;

TEST(DescriptorFactoryTest, CreateEngineConfigDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_NO_THROW(DescriptorFactory::create(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &descriptor));
    EXPECT_NE(descriptor, nullptr);

    ASSERT_NO_THROW(DescriptorFactory::destroy(descriptor));
}

TEST(DescriptorFactoryTest, CreateEngineDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_NO_THROW(DescriptorFactory::create(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, &descriptor));
    EXPECT_NE(descriptor, nullptr);

    ASSERT_NO_THROW(DescriptorFactory::destroy(descriptor));
}

TEST(DescriptorFactoryTest, CreateExecutionPlanDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_NO_THROW(
        DescriptorFactory::create(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &descriptor));
    EXPECT_NE(descriptor, nullptr);

    ASSERT_NO_THROW(DescriptorFactory::destroy(descriptor));
}

TEST(DescriptorFactoryTest, CreateGraphDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_NO_THROW(
        DescriptorFactory::create(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor));

    EXPECT_NE(descriptor, nullptr);

    ASSERT_NO_THROW(DescriptorFactory::destroy(descriptor));
}

TEST(DescriptorFactoryTest, CreateUnsupportedDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        DescriptorFactory::create(static_cast<hipdnnBackendDescriptorType_t>(999), &descriptor),
        HIPDNN_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, NullDescriptorPointer)
{
    ASSERT_THROW_HIPDNN_STATUS(
        DescriptorFactory::create(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(DescriptorFactoryTest, CreateGraphExtValidInput)
{
    auto builder = flatbuffer_test_utils::createValidGraph();
    auto serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_NO_THROW(DescriptorFactory::createGraphExt(
        &descriptor, serializedGraph.data(), serializedGraph.size()));

    EXPECT_NE(descriptor, nullptr);

    ASSERT_NO_THROW(DescriptorFactory::destroy(descriptor));
}

TEST(DescriptorFactoryTest, CreateGraphExtNullDescriptorPointer)
{
    auto builder = flatbuffer_test_utils::createValidGraph();
    auto serializedGraph = builder.Release();

    ASSERT_THROW_HIPDNN_STATUS(
        DescriptorFactory::createGraphExt(nullptr, serializedGraph.data(), serializedGraph.size()),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(DescriptorFactoryTest, CreateGraphExtNullSerializedGraph)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(DescriptorFactory::createGraphExt(&descriptor, nullptr, 10),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, CreateGraphExtZeroByteSize)
{
    auto builder = flatbuffer_test_utils::createValidGraph();
    auto serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        DescriptorFactory::createGraphExt(&descriptor, serializedGraph.data(), 0),
        HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, CreateGraphExtInvalidGraphData)
{
    const std::array<uint8_t, 2> invalidSerializedGraph = {0xFF, 0xFF};
    size_t graphByteSize = sizeof(invalidSerializedGraph);

    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(DescriptorFactory::createGraphExt(
                                   &descriptor, invalidSerializedGraph.data(), graphByteSize),
                               HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, TestHandleFactory)
{
    hipdnnHandle_t handleT = nullptr;

    ASSERT_NO_THROW(hipdnn_backend::HandleFactory::createHandle(&handleT));
    EXPECT_NE(handleT, nullptr);

    hipdnn_backend::HandleFactory::destroyHandle(handleT);
    handleT = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(hipdnn_backend::HandleFactory::destroyHandle(nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(hipdnn_backend::HandleFactory::createHandle(nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(DescriptorFactoryTest, CreateVariantDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    ASSERT_NO_THROW(DescriptorFactory::create(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &descriptor));
    EXPECT_NE(descriptor, nullptr);

    auto variantDescriptor = descriptor->asDescriptor<VariantDescriptor>();
    EXPECT_FALSE(variantDescriptor->isFinalized());

    ASSERT_NO_THROW(DescriptorFactory::destroy(descriptor));
}

TEST(DescriptorFactoryTest, CreateVariantPackWithNullDescriptor)
{
    ASSERT_THROW_HIPDNN_STATUS(
        DescriptorFactory::create(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(DescriptorFactoryTest, CreateVariantPackWithUnsupportedType)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(
        DescriptorFactory::create(static_cast<hipdnnBackendDescriptorType_t>(999), &descriptor),
        HIPDNN_STATUS_NOT_SUPPORTED);

    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, DestroyNull)
{
    ASSERT_THROW_HIPDNN_STATUS(DescriptorFactory::destroy(nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}
