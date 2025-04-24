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