// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "backend_descriptor.hpp"

namespace hipdnn_backend
{

class Execution_plan_descriptor : public hipdnnBackendDescriptor
{
private:
    hipdnnHandle_t _handle = nullptr;
    hipdnnBackendDescriptor_t _engine_config = nullptr;

    void get_workspace_size(hipdnnBackendAttributeType_t attribute_type,
                            int64_t requested_element_count,
                            int64_t* element_count,
                            void* array_of_elements);

    void set_handle(hipdnnBackendAttributeType_t attribute_type,
                    int64_t element_count,
                    const void* array_of_elements);

    void set_engine_config(hipdnnBackendAttributeType_t attribute_type,
                           int64_t element_count,
                           const void* array_of_elements);

public:
    Execution_plan_descriptor();
    ~Execution_plan_descriptor() override = default;

    void finalize() override;

    void get_attribute(hipdnnBackendAttributeName_t attribute_name,
                       hipdnnBackendAttributeType_t attribute_type,
                       int64_t requested_element_count,
                       int64_t* element_count,
                       void* array_of_elements) override;

    void set_attribute(hipdnnBackendAttributeName_t attribute_name,
                       hipdnnBackendAttributeType_t attribute_type,
                       int64_t element_count,
                       const void* array_of_elements) override;
};

} // namespace hipdnn_backend
