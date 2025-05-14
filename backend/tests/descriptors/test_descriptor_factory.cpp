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
#include "test_macros.hpp"

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
    ASSERT_THROW_HIPDNN_STATUS(
        Descriptor_factory::create(static_cast<hipdnnBackendDescriptorType_t>(999), &descriptor),
        HIPDNN_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, NullDescriptorPointer)
{
    ASSERT_THROW_HIPDNN_STATUS(
        Descriptor_factory::create(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
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

    ASSERT_THROW_HIPDNN_STATUS(Descriptor_factory::create_graph_ext(
                                   nullptr, serialized_graph.data(), serialized_graph.size()),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(DescriptorFactoryTest, CreateGraphExtNullSerializedGraph)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(Descriptor_factory::create_graph_ext(&descriptor, nullptr, 10),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, CreateGraphExtZeroByteSize)
{
    auto builder = flatbuffer_test_utils::create_valid_graph();
    auto serialized_graph = builder.Release();

    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        Descriptor_factory::create_graph_ext(&descriptor, serialized_graph.data(), 0),
        HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, CreateGraphExtInvalidGraphData)
{
    const std::array<uint8_t, 2> invalid_serialized_graph = {0xFF, 0xFF};
    size_t graph_byte_size = sizeof(invalid_serialized_graph);

    hipdnnBackendDescriptor_t descriptor = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(Descriptor_factory::create_graph_ext(
                                   &descriptor, invalid_serialized_graph.data(), graph_byte_size),
                               HIPDNN_STATUS_BAD_PARAM);

    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, TestHandleFactory)
{
    hipdnnHandle_t handle_t = nullptr;

    ASSERT_NO_THROW(hipdnn_backend::Handle_factory::create_handle(&handle_t));
    EXPECT_NE(handle_t, nullptr);

    ASSERT_THROW_HIPDNN_STATUS(hipdnn_backend::Handle_factory::create_handle(nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
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
    ASSERT_THROW_HIPDNN_STATUS(
        Descriptor_factory::create(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(Descriptor_Factory_Test, Create_Variant_Pack_With_Unsupported_Type)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(
        Descriptor_factory::create(static_cast<hipdnnBackendDescriptorType_t>(999), &descriptor),
        HIPDNN_STATUS_NOT_SUPPORTED);

    EXPECT_EQ(descriptor, nullptr);
}
