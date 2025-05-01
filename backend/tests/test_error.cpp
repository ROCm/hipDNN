// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "error.hpp"
#include <gtest/gtest.h>

using namespace hipdnn_backend;

TEST(ErrorTests, SetLastError)
{
    // Test setting a valid error
    hipdnnStatus_t status = set_last_error(HIPDNN_STATUS_NOT_SUPPORTED, "Feature not supported");
    EXPECT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);

    // Test setting a success status
    status = set_last_error(HIPDNN_STATUS_SUCCESS, "Operation successful");
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST(ErrorTests, GetBackendDescriptorTypeName)
{
    EXPECT_STREQ(hipdnn_get_backend_descriptor_type_name(HIPDNN_BACKEND_ENGINE_DESCRIPTOR),
                 "HIPDNN_BACKEND_ENGINE_DESCRIPTOR");
    EXPECT_STREQ(hipdnn_get_backend_descriptor_type_name(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR),
                 "HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR");
    EXPECT_STREQ(hipdnn_get_backend_descriptor_type_name(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR),
                 "HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR");
    EXPECT_STREQ(hipdnn_get_backend_descriptor_type_name(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR),
                 "HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR");
    EXPECT_STREQ(
        hipdnn_get_backend_descriptor_type_name(HIPDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR),
        "HIPDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR");
    EXPECT_STREQ(hipdnn_get_backend_descriptor_type_name(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR),
                 "HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR");
    EXPECT_STREQ(hipdnn_get_backend_descriptor_type_name(HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR),
                 "HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR");
    EXPECT_STREQ(hipdnn_get_backend_descriptor_type_name(HIPDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR),
                 "HIPDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR");
    EXPECT_STREQ(
        hipdnn_get_backend_descriptor_type_name(HIPDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR),
        "HIPDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR");
    EXPECT_STREQ(hipdnn_get_backend_descriptor_type_name(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR),
                 "HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR");
    EXPECT_STREQ(hipdnn_get_backend_descriptor_type_name(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR),
                 "HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR");
    EXPECT_STREQ(hipdnn_get_backend_descriptor_type_name(HIPDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR),
                 "HIPDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR");
    EXPECT_STREQ(hipdnn_get_backend_descriptor_type_name(
                     HIPDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR),
                 "HIPDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR");

    // Test unknown type
    EXPECT_STREQ(
        hipdnn_get_backend_descriptor_type_name(static_cast<hipdnnBackendDescriptorType_t>(-1)),
        "UNKNOWN_TYPE");
}

TEST(ErrorTests, GetBackendAttributeName)
{
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINEHEUR_MODE),
                 "HIPDNN_ATTR_ENGINEHEUR_MODE");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH),
                 "HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINEHEUR_RESULTS),
                 "HIPDNN_ATTR_ENGINEHEUR_RESULTS");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET),
                 "HIPDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP),
                 "HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP");

    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINECFG_ENGINE),
                 "HIPDNN_ATTR_ENGINECFG_ENGINE");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO),
                 "HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES),
                 "HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE),
                 "HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE");

    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_EXECUTION_PLAN_HANDLE),
                 "HIPDNN_ATTR_EXECUTION_PLAN_HANDLE");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG),
                 "HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE),
                 "HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE");
    EXPECT_STREQ(
        hipdnn_get_attribute_name_string(HIPDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS),
        "HIPDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS");
    EXPECT_STREQ(
        hipdnn_get_attribute_name_string(HIPDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS),
        "HIPDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION),
                 "HIPDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE),
                 "HIPDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_EXECUTION_PLAN_DEVICEPROP),
                 "HIPDNN_ATTR_EXECUTION_PLAN_DEVICEPROP");

    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID),
                 "HIPDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_INTERMEDIATE_INFO_SIZE),
                 "HIPDNN_ATTR_INTERMEDIATE_INFO_SIZE");
    EXPECT_STREQ(
        hipdnn_get_attribute_name_string(HIPDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS),
        "HIPDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS");
    EXPECT_STREQ(
        hipdnn_get_attribute_name_string(HIPDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES),
        "HIPDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES");

    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE),
                 "HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE),
                 "HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE");

    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE),
                 "HIPDNN_ATTR_OPERATIONGRAPH_HANDLE");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_OPERATIONGRAPH_OPS),
                 "HIPDNN_ATTR_OPERATIONGRAPH_OPS");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT),
                 "HIPDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT");
    EXPECT_STREQ(
        hipdnn_get_attribute_name_string(HIPDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED),
        "HIPDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED");

    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS),
                 "HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS),
                 "HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_VARIANT_PACK_INTERMEDIATES),
                 "HIPDNN_ATTR_VARIANT_PACK_INTERMEDIATES");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_VARIANT_PACK_WORKSPACE),
                 "HIPDNN_ATTR_VARIANT_PACK_WORKSPACE");

    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_LAYOUT_INFO_TENSOR_UID),
                 "HIPDNN_ATTR_LAYOUT_INFO_TENSOR_UID");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_LAYOUT_INFO_TYPES),
                 "HIPDNN_ATTR_LAYOUT_INFO_TYPES");

    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_KNOB_INFO_TYPE),
                 "HIPDNN_ATTR_KNOB_INFO_TYPE");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE),
                 "HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE),
                 "HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_KNOB_INFO_STRIDE),
                 "HIPDNN_ATTR_KNOB_INFO_STRIDE");

    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH),
                 "HIPDNN_ATTR_ENGINE_OPERATION_GRAPH");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINE_GLOBAL_INDEX),
                 "HIPDNN_ATTR_ENGINE_GLOBAL_INDEX");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINE_KNOB_INFO),
                 "HIPDNN_ATTR_ENGINE_KNOB_INFO");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE),
                 "HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINE_LAYOUT_INFO),
                 "HIPDNN_ATTR_ENGINE_LAYOUT_INFO");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE),
                 "HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINE_SM_COUNT_TARGET),
                 "HIPDNN_ATTR_ENGINE_SM_COUNT_TARGET");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_ENGINE_DEVICEPROP),
                 "HIPDNN_ATTR_ENGINE_DEVICEPROP");

    EXPECT_STREQ(
        hipdnn_get_attribute_name_string(HIPDNN_ATTR_KERNEL_CACHE_IS_ENGINECFG_KERNEL_CACHED),
        "HIPDNN_ATTR_KERNEL_CACHE_IS_ENGINECFG_KERNEL_CACHED");

    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_DEVICEPROP_DEVICE_ID),
                 "HIPDNN_ATTR_DEVICEPROP_DEVICE_ID");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_DEVICEPROP_HANDLE),
                 "HIPDNN_ATTR_DEVICEPROP_HANDLE");
    EXPECT_STREQ(hipdnn_get_attribute_name_string(HIPDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION),
                 "HIPDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION");
}