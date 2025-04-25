// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

TEST(HipDNNBackendTest, WillCreateDestroyGraphDescriptorSuccessfully)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnBackendDestroyDescriptor(descriptor);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST(HipDNNBackendTest, WillNotCreateDescriptorIfPassedNullptr)
{
    hipdnnStatus_t status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, nullptr);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

TEST(HipDNNBackendTest, WillNotCreateDescriptorIfTypeNotSupported)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t status
        = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, &descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST(HipDNNBackendTest, WontDestroyDescriptorIfNull)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t status = hipdnnBackendDestroyDescriptor(descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

TEST(HipDNNBackendTest, Execute)
{
    hipdnnHandle_t            handle         = nullptr;
    hipdnnBackendDescriptor_t execution_plan = nullptr;
    hipdnnBackendDescriptor_t variant_pack   = nullptr;

    hipdnnStatus_t status = hipdnnBackendExecute(handle, execution_plan, variant_pack);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

TEST(HipDNNBackendTest, Finalize)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    hipdnnStatus_t status = hipdnnBackendFinalize(descriptor);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

TEST(HipDNNBackendTest, GetAttribute)
{
    hipdnnBackendDescriptor_t    descriptor              = nullptr;
    hipdnnBackendAttributeName_t attribute_name          = HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
    hipdnnBackendAttributeType_t attribute_type          = HIPDNN_TYPE_NUMERICAL_NOTE;
    int64_t                      requested_element_count = 0;
    int64_t                      element_count           = 0;
    void*                        array_of_elements       = nullptr;

    hipdnnStatus_t status = hipdnnBackendGetAttribute(descriptor,
                                                      attribute_name,
                                                      attribute_type,
                                                      requested_element_count,
                                                      &element_count,
                                                      array_of_elements);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

TEST(HipDNNBackendTest, SetAttribute)
{
    hipdnnBackendDescriptor_t    descriptor        = nullptr;
    hipdnnBackendAttributeName_t attribute_name    = HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
    hipdnnBackendAttributeType_t attribute_type    = HIPDNN_TYPE_NUMERICAL_NOTE;
    int64_t                      element_count     = 0;
    void*                        array_of_elements = nullptr;

    hipdnnStatus_t status = hipdnnBackendSetAttribute(
        descriptor, attribute_name, attribute_type, element_count, array_of_elements);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
}

TEST(HipDNNBackendTest, WillSetBackendGraphCorrectly)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
                                                                       tensor_attributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto                                                               graph
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "Test GRAPH!",
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      &tensor_attributes,
                                                      &nodes);
    builder.Finish(graph);
    flatbuffers::DetachedBuffer serialized_graph = builder.Release();

    hipdnnBackendDescriptor_t descriptor = nullptr;

    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &descriptor, serialized_graph.data(), serialized_graph.size());

    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
    hipdnnBackendDestroyDescriptor(descriptor);
}

TEST(HipDNNBackendTest, WillFailToCreateGraphIfGraphIsNull)
{
    hipdnnBackendDescriptor_t descriptor = nullptr;

    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(&descriptor, nullptr, 0);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(descriptor, nullptr);
}
