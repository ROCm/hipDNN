// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/descriptor_factory.hpp"
#include "descriptors/engine_config_descriptor.hpp"
#include "descriptors/engine_descriptor.hpp"
#include "descriptors/execution_plan_descriptor.hpp"
#include "descriptors/graph_descriptor.hpp"
#include "descriptors/variant_descriptor.hpp"
#include "flatbuffer_test_utils.hpp"
#include "handle/handle_factory.hpp"
#include "hipdnn_exception.hpp"

#include "gtest/gtest.h"

using namespace hipdnn_backend;

TEST(DescriptorFactoryTest, CreateEngineConfigDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_NO_THROW(Descriptor_factory::create(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &descriptor));
    EXPECT_NE(descriptor, nullptr);

    delete descriptor;
}

TEST(DescriptorFactoryTest, CreateEngineDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_NO_THROW(Descriptor_factory::create(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, &descriptor));
    EXPECT_NE(descriptor, nullptr);

    delete descriptor;
}

TEST(DescriptorFactoryTest, CreateExecutionPlanDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_NO_THROW(
        Descriptor_factory::create(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &descriptor));
    EXPECT_NE(descriptor, nullptr);

    delete descriptor;
}

TEST(DescriptorFactoryTest, CreateGraphDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_NO_THROW(
        Descriptor_factory::create(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor));

    EXPECT_NE(descriptor, nullptr);

    delete descriptor;
}

TEST(DescriptorFactoryTest, CreateUnsupportedDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_THROW(
        Descriptor_factory::create(static_cast<hipdnnBackendDescriptorType_t>(999), &descriptor),
        hipdnn_backend::Hipdnn_exception);
    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, NullDescriptorPointer)
{
    ASSERT_THROW(Descriptor_factory::create(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, nullptr),
                 hipdnn_backend::Hipdnn_exception);
}

TEST(DescriptorFactoryTest, CreateGraphExtValidInput)
{
    auto builder = flatbuffer_test_utils::create_valid_graph();
    auto serialized_graph = builder.Release();

    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_NO_THROW(Descriptor_factory::create_graph_ext(
        &descriptor, serialized_graph.data(), serialized_graph.size()));

    EXPECT_NE(descriptor, nullptr);

    delete descriptor;
}

TEST(DescriptorFactoryTest, CreateGraphExtNullDescriptorPointer)
{
    auto builder = flatbuffer_test_utils::create_valid_graph();
    auto serialized_graph = builder.Release();

    ASSERT_THROW(Descriptor_factory::create_graph_ext(
                     nullptr, serialized_graph.data(), serialized_graph.size()),
                 hipdnn_backend::Hipdnn_exception);
}

TEST(DescriptorFactoryTest, CreateGraphExtNullSerializedGraph)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_THROW(Descriptor_factory::create_graph_ext(&descriptor, nullptr, 10),
                 hipdnn_backend::Hipdnn_exception);

    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, CreateGraphExtZeroByteSize)
{
    auto builder = flatbuffer_test_utils::create_valid_graph();
    auto serialized_graph = builder.Release();

    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_THROW(Descriptor_factory::create_graph_ext(&descriptor, serialized_graph.data(), 0),
                 hipdnn_backend::Hipdnn_exception);

    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, CreateGraphExtInvalidGraphData)
{
    const std::array<uint8_t, 2> invalid_serialized_graph = {0xFF, 0xFF};
    size_t graph_byte_size = sizeof(invalid_serialized_graph);

    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_THROW(Descriptor_factory::create_graph_ext(
                     &descriptor, invalid_serialized_graph.data(), graph_byte_size),
                 hipdnn_backend::Hipdnn_exception);

    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, TestHandleFactory)
{
    hipdnnHandle_t handle_t = nullptr;

    ASSERT_NO_THROW(hipdnn_backend::Handle_factory::create_handle(&handle_t));
    EXPECT_NE(handle_t, nullptr);

    ASSERT_THROW(hipdnn_backend::Handle_factory::create_handle(nullptr),
                 hipdnn_backend::Hipdnn_exception);
}

TEST(Descriptor_Factory_Test, Create_Variant_Descriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    ASSERT_NO_THROW(
        Descriptor_factory::create(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &descriptor));
    EXPECT_NE(descriptor, nullptr);

    auto variant_descriptor = dynamic_cast<Variant_descriptor*>(descriptor);
    EXPECT_NE(variant_descriptor, nullptr);

    EXPECT_FALSE(variant_descriptor->is_finalized());

    delete variant_descriptor;
}

TEST(Descriptor_Factory_Test, Create_Variant_Pack_With_Null_Descriptor)
{
    ASSERT_THROW(Descriptor_factory::create(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, nullptr),
                 hipdnn_backend::Hipdnn_exception);
}

TEST(Descriptor_Factory_Test, Create_Variant_Pack_With_Unsupported_Type)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    ASSERT_THROW(
        Descriptor_factory::create(static_cast<hipdnnBackendDescriptorType_t>(999), &descriptor),
        hipdnn_backend::Hipdnn_exception);

    EXPECT_EQ(descriptor, nullptr);
}
