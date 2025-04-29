// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/descriptor_factory.hpp"
#include "descriptors/graph_descriptor.hpp"
#include "flatbuffer_test_utils.hpp"
#include "handle/handle_factory.hpp"

#include "gtest/gtest.h"

using namespace hipdnn_backend;

TEST(DescriptorFactoryTest, CreateGraphDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    auto status = Descriptor_factory::create(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_NE(descriptor, nullptr);

    delete static_cast<Graph_descriptor*>(descriptor);
}

TEST(DescriptorFactoryTest, CreateUnsupportedDescriptor)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    auto                      status
        = Descriptor_factory::create(static_cast<hipdnnBackendDescriptorType_t>(999), &descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, NullDescriptorPointer)
{
    auto status = Descriptor_factory::create(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, nullptr);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

TEST(DescriptorFactoryTest, CreateGraphExtValidInput)
{
    auto builder          = flatbuffer_test_utils::create_valid_graph();
    auto serialized_graph = builder.Release();

    hipdnnBackendDescriptor_t descriptor = nullptr;
    auto                      status     = Descriptor_factory::create_graph_ext(
        &descriptor, serialized_graph.data(), serialized_graph.size());

    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_NE(descriptor, nullptr);

    delete static_cast<Graph_descriptor*>(descriptor);
}

TEST(DescriptorFactoryTest, CreateGraphExtNullDescriptorPointer)
{
    auto builder          = flatbuffer_test_utils::create_valid_graph();
    auto serialized_graph = builder.Release();

    auto status = Descriptor_factory::create_graph_ext(
        nullptr, serialized_graph.data(), serialized_graph.size());

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

TEST(DescriptorFactoryTest, CreateGraphExtNullSerializedGraph)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    auto status = Descriptor_factory::create_graph_ext(&descriptor, nullptr, 10);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, CreateGraphExtZeroByteSize)
{
    auto builder          = flatbuffer_test_utils::create_valid_graph();
    auto serialized_graph = builder.Release();

    hipdnnBackendDescriptor_t descriptor = nullptr;
    auto status = Descriptor_factory::create_graph_ext(&descriptor, serialized_graph.data(), 0);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, CreateGraphExtInvalidGraphData)
{
    const std::array<uint8_t, 2> invalid_serialized_graph = {0xFF, 0xFF};
    size_t                       graph_byte_size          = sizeof(invalid_serialized_graph);

    hipdnnBackendDescriptor_t descriptor = nullptr;
    auto                      status     = Descriptor_factory::create_graph_ext(
        &descriptor, invalid_serialized_graph.data(), graph_byte_size);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(descriptor, nullptr);
}

TEST(DescriptorFactoryTest, TestHandleFactory)
{
    hipdnnHandle_t handle_t = nullptr;

    auto status = hipdnn_backend::Handle_factory::create_handle(&handle_t);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
    EXPECT_NE(handle_t, nullptr);

    status = hipdnn_backend::Handle_factory::create_handle(nullptr);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}
