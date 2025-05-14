// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "backend_descriptor.hpp"

namespace hipdnn_backend
{

class Engine_descriptor;

class Engine_config_descriptor : public hipdnnBackendDescriptor
{
private:
    const Engine_descriptor* _engine = nullptr;
    int64_t _max_workspace_size = INVALID_WORKSPACE_SIZE;

    void set_engine(hipdnnBackendAttributeType_t attribute_type,
                    int64_t element_count,
                    const void* array_of_elements);

    void get_engine(hipdnnBackendAttributeType_t attribute_type,
                    int64_t requested_element_count,
                    int64_t* element_count,
                    void* array_of_elements);

    void get_max_workspace_size(hipdnnBackendAttributeType_t attribute_type,
                                int64_t requested_element_count,
                                int64_t* element_count,
                                void* array_of_elements) const;

public:
    static constexpr int64_t INVALID_WORKSPACE_SIZE = -1;

    Engine_config_descriptor();
    ~Engine_config_descriptor() override = default;

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

    void set_max_workspace_size(int64_t max_workspace_size);
};

} // namespace hipdnn_backend
