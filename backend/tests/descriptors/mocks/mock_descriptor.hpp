// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "descriptors/backend_descriptor.hpp"

#include <gmock/gmock.h>

namespace hipdnn_backend
{

class Mock_descriptor : public hipdnnBackendDescriptor
{
public:
    Mock_descriptor(hipdnnBackendDescriptorType_t desc_type = HIPDNN_INVALID_TYPE)
    {
        type = desc_type;
    }
    ~Mock_descriptor() override = default;

    MOCK_METHOD(void, finalize, (), (override));
    MOCK_METHOD(bool, is_finalized, (), (const, override));
    MOCK_METHOD(void,
                set_attribute,
                (hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t element_count,
                 const void* array_of_element),
                (override));
    MOCK_METHOD(void,
                get_attribute,
                (hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t requested_element_count,
                 int64_t* element_count,
                 void* array_of_elements),
                (override));
};

ACTION_P(SetArg4ToInt64, value) // NOLINT
{
    *static_cast<int64_t*>(arg4) = value;
}

} // namespace hipdnn_backend
