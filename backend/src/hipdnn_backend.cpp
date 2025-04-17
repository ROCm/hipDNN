// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <iostream>

hipdnnStatus_t hipdnnCreate(hipdnnHandle_t* handle)
{
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle)
{
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipStream_t streamId)
{
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnBackendCreateDescriptor(hipdnnBackendDescriptorType_t descriptor_type,
                                             hipdnnBackendDescriptor_t*    descriptor)
{
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnBackendDestroyDescriptor(hipdnnBackendDescriptor_t descriptor)
{
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnBackendExecute(hipdnnHandle_t            handle,
                                    hipdnnBackendDescriptor_t execution_plan,
                                    hipdnnBackendDescriptor_t variant_pack)
{
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnBackendFinalize(hipdnnBackendDescriptor_t descriptor)
{
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnBackendGetAttribute(hipdnnBackendDescriptor_t    descriptor,
                                         hipdnnBackendAttributeName_t attribute_name,
                                         hipdnnBackendAttributeType_t attribute_type,
                                         int64_t                      requested_element_count,
                                         int64_t*                     element_count,
                                         void*                        array_of_elements)
{
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnBackendSetAttribute(hipdnnBackendDescriptor_t    descriptor,
                                         hipdnnBackendAttributeName_t attribute_name,
                                         hipdnnBackendAttributeType_t attribute_type,
                                         int64_t                      element_count,
                                         void*                        array_of_elements)
{
    return HIPDNN_STATUS_SUCCESS;
}
