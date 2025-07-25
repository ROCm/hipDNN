// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "backend_descriptor.hpp"

namespace hipdnn_backend
{

class Engine_config_descriptor;
class Execution_plan_descriptor : public hipdnnBackendDescriptorImpl<Execution_plan_descriptor>
{
private:
    hipdnnHandle_t _handle = nullptr;
    std::shared_ptr<const Engine_config_descriptor> _engine_config;

    void get_workspace_size(hipdnnBackendAttributeType_t attribute_type,
                            int64_t requested_element_count,
                            int64_t* element_count,
                            void* array_of_elements) const;

    void set_handle(hipdnnBackendAttributeType_t attribute_type,
                    int64_t element_count,
                    const void* array_of_elements);

    void set_engine_config(hipdnnBackendAttributeType_t attribute_type,
                           int64_t element_count,
                           const void* array_of_elements);

    void get_engine_config(hipdnnBackendAttributeType_t attribute_type,
                           int64_t requested_element_count,
                           int64_t* element_count,
                           void* array_of_elements) const;

public:
    void finalize() override;

    void get_attribute(hipdnnBackendAttributeName_t attribute_name,
                       hipdnnBackendAttributeType_t attribute_type,
                       int64_t requested_element_count,
                       int64_t* element_count,
                       void* array_of_elements) const override;

    void set_attribute(hipdnnBackendAttributeName_t attribute_name,
                       hipdnnBackendAttributeType_t attribute_type,
                       int64_t element_count,
                       const void* array_of_elements) override;

    // Throws an exception if the descriptor is not finalized.
    std::shared_ptr<const Engine_config_descriptor> get_engine_config() const;

    static hipdnnBackendDescriptorType_t get_static_type();
};

} // namespace hipdnn_backend
