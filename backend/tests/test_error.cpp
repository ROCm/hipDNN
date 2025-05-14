// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "error.hpp"
#include <gtest/gtest.h>

using namespace hipdnn_backend;

TEST(ErrorTests, StaticSetLastError)
{
    // Test setting a success status
    hipdnnStatus_t status
        = Last_error_manager::set_last_error(HIPDNN_STATUS_SUCCESS, "Operation successful");
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    // Test setting a valid error
    std::string error_message = "An error occurred";
    status = Last_error_manager::set_last_error(HIPDNN_STATUS_NOT_SUPPORTED, error_message.c_str());
    EXPECT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);
    EXPECT_STREQ(Last_error_manager::get_last_error(), error_message.c_str());
}

TEST(ErrorTests, ErrorCStringMessagePerThread)
{
    std::string main_error = "Main thread error";
    std::string worker_error = "Worker thread error";
    Last_error_manager::set_last_error(HIPDNN_STATUS_BAD_PARAM, main_error.c_str());

    std::string thread_error;
    std::thread t([&thread_error, worker_error]() {
        Last_error_manager::set_last_error(HIPDNN_STATUS_NOT_SUPPORTED, worker_error.c_str());
        thread_error = Last_error_manager::get_last_error();
    });
    t.join();

    EXPECT_EQ(Last_error_manager::get_last_error(), main_error);
    EXPECT_EQ(thread_error, worker_error);
}

TEST(ErrorTests, ErrorSTDStringMessagePerThread)
{
    std::string main_error = "Main thread error";
    std::string worker_error = "Worker thread error";
    Last_error_manager::set_last_error(HIPDNN_STATUS_BAD_PARAM, main_error);

    std::string thread_error;
    std::thread t([&thread_error, worker_error]() {
        Last_error_manager::set_last_error(HIPDNN_STATUS_NOT_SUPPORTED, worker_error);
        thread_error = Last_error_manager::get_last_error();
    });
    t.join();

    EXPECT_EQ(Last_error_manager::get_last_error(), main_error);
    EXPECT_EQ(thread_error, worker_error);
}

TEST(ErrorTests, SetSuccessSTDStringDoesNotSetErrorMessage)
{
    std::string error_message = "This message should not be set";
    hipdnnStatus_t status
        = Last_error_manager::set_last_error(HIPDNN_STATUS_SUCCESS, error_message);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    EXPECT_NE(Last_error_manager::get_last_error(), error_message);
}

TEST(ErrorTests, SetSuccessCStringDoesNotSetErrorMessage)
{
    std::string error_message = "This message should not be set";
    hipdnnStatus_t status
        = Last_error_manager::set_last_error(HIPDNN_STATUS_SUCCESS, error_message.c_str());
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    EXPECT_NE(Last_error_manager::get_last_error(), error_message);
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

TEST(ErrorTests, Get_Status_String)
{
    EXPECT_STREQ(hipdnn_get_status_string(HIPDNN_STATUS_SUCCESS), "HIPDNN_STATUS_SUCCESS");
    EXPECT_STREQ(hipdnn_get_status_string(HIPDNN_STATUS_NOT_INITIALIZED),
                 "HIPDNN_STATUS_NOT_INITIALIZED");
    EXPECT_STREQ(hipdnn_get_status_string(HIPDNN_STATUS_BAD_PARAM), "HIPDNN_STATUS_BAD_PARAM");
    EXPECT_STREQ(hipdnn_get_status_string(HIPDNN_STATUS_BAD_PARAM_NULL_POINTER),
                 "HIPDNN_STATUS_BAD_PARAM_NULL_POINTER");
    EXPECT_STREQ(hipdnn_get_status_string(HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED),
                 "HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED");
    EXPECT_STREQ(hipdnn_get_status_string(HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND),
                 "HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND");
    EXPECT_STREQ(hipdnn_get_status_string(HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT),
                 "HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT");
    EXPECT_STREQ(hipdnn_get_status_string(HIPDNN_STATUS_BAD_PARAM_STREAM_MISMATCH),
                 "HIPDNN_STATUS_BAD_PARAM_STREAM_MISMATCH");
    EXPECT_STREQ(hipdnn_get_status_string(HIPDNN_STATUS_NOT_SUPPORTED),
                 "HIPDNN_STATUS_NOT_SUPPORTED");
    EXPECT_STREQ(hipdnn_get_status_string(HIPDNN_STATUS_INTERNAL_ERROR),
                 "HIPDNN_STATUS_INTERNAL_ERROR");
    EXPECT_STREQ(hipdnn_get_status_string(HIPDNN_STATUS_ALLOC_FAILED),
                 "HIPDNN_STATUS_ALLOC_FAILED");
    EXPECT_STREQ(hipdnn_get_status_string(HIPDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED),
                 "HIPDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED");
    EXPECT_STREQ(hipdnn_get_status_string(HIPDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED),
                 "HIPDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED");
    EXPECT_STREQ(hipdnn_get_status_string(HIPDNN_STATUS_EXECUTION_FAILED),
                 "HIPDNN_STATUS_EXECUTION_FAILED");

    EXPECT_STREQ(hipdnn_get_status_string(static_cast<hipdnnStatus_t>(-1)),
                 "HIPDNN_STATUS_UNKNOWN");
}

TEST(ErrorTests, Get_Attribute_Type_String)
{
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_HANDLE), "HIPDNN_TYPE_HANDLE");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_DATA_TYPE), "HIPDNN_TYPE_DATA_TYPE");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_BOOLEAN), "HIPDNN_TYPE_BOOLEAN");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_INT64), "HIPDNN_TYPE_INT64");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_FLOAT), "HIPDNN_TYPE_FLOAT");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_DOUBLE), "HIPDNN_TYPE_DOUBLE");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_VOID_PTR), "HIPDNN_TYPE_VOID_PTR");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_HEUR_MODE), "HIPDNN_TYPE_HEUR_MODE");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_KNOB_TYPE), "HIPDNN_TYPE_KNOB_TYPE");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_NAN_PROPOGATION),
                 "HIPDNN_TYPE_NAN_PROPOGATION");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_NUMERICAL_NOTE),
                 "HIPDNN_TYPE_NUMERICAL_NOTE");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_LAYOUT_TYPE),
                 "HIPDNN_TYPE_LAYOUT_TYPE");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_ATTRIB_NAME),
                 "HIPDNN_TYPE_ATTRIB_NAME");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_BACKEND_DESCRIPTOR),
                 "HIPDNN_TYPE_BACKEND_DESCRIPTOR");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_GENSTATS_MODE),
                 "HIPDNN_TYPE_GENSTATS_MODE");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_BN_FINALIZE_STATS_MODE),
                 "HIPDNN_TYPE_BN_FINALIZE_STATS_MODE");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_BEHAVIOR_NOTE),
                 "HIPDNN_TYPE_BEHAVIOR_NOTE");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_TENSOR_REORDERING_MODE),
                 "HIPDNN_TYPE_TENSOR_REORDERING_MODE");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_INT32), "HIPDNN_TYPE_INT32");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_CHAR), "HIPDNN_TYPE_CHAR");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_SIGNAL_MODE),
                 "HIPDNN_TYPE_SIGNAL_MODE");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_FRACTION), "HIPDNN_TYPE_FRACTION");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_NORM_FWD_PHASE),
                 "HIPDNN_TYPE_NORM_FWD_PHASE");
    EXPECT_STREQ(hipdnn_get_attribute_type_string(HIPDNN_TYPE_RNG_DISTRIBUTION),
                 "HIPDNN_TYPE_RNG_DISTRIBUTION");

    EXPECT_STREQ(hipdnn_get_attribute_type_string(static_cast<hipdnnBackendAttributeType_t>(-1)),
                 "HIPDNN_ATTRIBUTE_UNKNOWN");
}