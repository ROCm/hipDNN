// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "../test_plugins/TestPluginConstants.hpp"
#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include <array>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <vector>

using namespace hipdnn_tests::plugin_constants;
using namespace hipdnn_sdk::utilities;
namespace fs = std::filesystem;

TEST(IntegrationBackendDescriptor, CreateAndDestroy)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendDestroyDescriptor(descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationBackendDescriptor, CreateWithNullptr)
{
    hipdnnStatus_t status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, nullptr);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(IntegrationBackendDescriptor, WillNotCreateDescriptorIfTypeNotSupported)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t status = hipdnnBackendCreateDescriptor(HIPDNN_INVALID_TYPE, &descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST(IntegrationBackendDescriptor, WontDestroyDescriptorIfNull)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t status = hipdnnBackendDestroyDescriptor(descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(IntegrationBackendDescriptor, Finalize)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t status = hipdnnBackendFinalize(descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(IntegrationBackendDescriptor, GetAttribute)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    hipdnnBackendAttributeName_t attributeName = HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
    hipdnnBackendAttributeType_t attributeType = HIPDNN_TYPE_NUMERICAL_NOTE;
    int64_t requestedElementCount = 0;
    int64_t elementCount = 0;
    void* arrayOfElements = nullptr;

    hipdnnStatus_t status = hipdnnBackendGetAttribute(descriptor,
                                                      attributeName,
                                                      attributeType,
                                                      requestedElementCount,
                                                      &elementCount,
                                                      arrayOfElements);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(IntegrationBackendDescriptor, SetAttribute)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    hipdnnBackendAttributeName_t attributeName = HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
    hipdnnBackendAttributeType_t attributeType = HIPDNN_TYPE_NUMERICAL_NOTE;
    int64_t elementCount = 0;
    void* arrayOfElements = nullptr;

    hipdnnStatus_t status = hipdnnBackendSetAttribute(
        descriptor, attributeName, attributeType, elementCount, arrayOfElements);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(IntegrationBackendDescriptor, SetOperationGraph)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto graph
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "Test GRAPH!",
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graph);
    flatbuffers::DetachedBuffer serializedGraph = builder.Release();

    hipdnnBackendDescriptor_t descriptor = nullptr;

    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &descriptor, serializedGraph.data(), serializedGraph.size());

    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendSetAttribute(
        descriptor, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendFinalize(descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDestroyDescriptor(descriptor);
    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationBackendDescriptor, FinalizeInvalidOperationGraph)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;
    auto status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendFinalize(descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);

    status = hipdnnBackendDestroyDescriptor(descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

/// ANOTHER FILE vvv
