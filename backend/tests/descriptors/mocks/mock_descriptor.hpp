// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "descriptors/backend_descriptor.hpp"

#include <map>
#include <vector>

namespace hipdnn_backend
{

class Mock_descriptor : public Backend_descriptor
{
private:
    std::map<std::pair<hipdnnBackendAttributeName_t, hipdnnBackendAttributeType_t>,
             std::vector<const void*>>
        _attributes;

public:
    Mock_descriptor(hipdnnBackendDescriptorType_t desc_type = HIPDNN_INVALID_TYPE,
                    bool                          finalized = false);
    ~Mock_descriptor() override = default;

    hipdnnStatus_t set_data(hipdnnBackendAttributeName_t attribute_name,
                            hipdnnBackendAttributeType_t attribute_type,
                            int64_t                      element_count,
                            const void*                  elements);

    hipdnnStatus_t finalize() override;

    hipdnnStatus_t get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                 hipdnnBackendAttributeType_t attribute_type,
                                 int64_t                      requested_element_count,
                                 int64_t*                     element_count,
                                 void*                        array_of_elements) override;

    hipdnnStatus_t set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                 hipdnnBackendAttributeType_t attribute_type,
                                 int64_t                      element_count,
                                 const void*                  array_of_elements) override;
};

} // namespace hipdnn_backend
