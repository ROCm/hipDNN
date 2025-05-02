// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "backend_descriptor.hpp"

namespace hipdnn_backend
{

class Engine_descriptor : public hipdnnBackendDescriptor
{
private:
    hipdnnBackendDescriptor_t _graph = nullptr;

    void set_graph(hipdnnBackendAttributeType_t attribute_type,
                   int64_t element_count,
                   const void* array_of_elements);

public:
    Engine_descriptor();
    ~Engine_descriptor() override = default;

    void finalize() override;

    hipdnnStatus_t get_attribute(hipdnnBackendAttributeName_t attribute_name,
                                 hipdnnBackendAttributeType_t attribute_type,
                                 int64_t requested_element_count,
                                 int64_t* element_count,
                                 void* array_of_elements) override;

    hipdnnStatus_t set_attribute(hipdnnBackendAttributeName_t attribute_name,
                                 hipdnnBackendAttributeType_t attribute_type,
                                 int64_t element_count,
                                 const void* array_of_elements) override;
};

} // namespace hipdnn_backend
