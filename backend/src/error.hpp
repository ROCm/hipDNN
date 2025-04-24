// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipdnn_backend_descriptor_type.h"
#include "hipdnn_status.h"

namespace hipdnn_backend
{

inline hipdnnStatus_t set_last_error(hipdnnStatus_t status, const char* /*message*/)
{
    //TODO
    // Set the last error message
    // stash the message in some global theadsafe map.
    return status;
}

inline const char* hipdnn_get_backend_descriptor_type_name(hipdnnBackendDescriptorType_t type)
{
    switch(type)
    {
    case HIPDNN_BACKEND_ENGINE_DESCRIPTOR:
        return "HIPDNN_BACKEND_ENGINE_DESCRIPTOR";
    case HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR:
        return "HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR";
    case HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR:
        return "HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR";
    case HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
        return "HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR";
    case HIPDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR:
        return "HIPDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR";
    case HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR:
        return "HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR";
    case HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR:
        return "HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR";
    case HIPDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR:
        return "HIPDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR";
    case HIPDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR:
        return "HIPDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR";
    case HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
        return "HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR";
    case HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
        return "HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR";
    case HIPDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR:
        return "HIPDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR";
    case HIPDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR:
        return "HIPDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR";
    default:
        return "UNKNOWN_TYPE";
    }
}

}