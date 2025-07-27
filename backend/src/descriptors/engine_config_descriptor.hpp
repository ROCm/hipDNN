// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "backend_descriptor.hpp"
#include <flatbuffers/detached_buffer.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace hipdnn_backend
{

class Engine_descriptor;

class Engine_config_descriptor : public hipdnnBackendDescriptorImpl<Engine_config_descriptor>
{
private:
    std::shared_ptr<const Engine_descriptor> _engine;
    int64_t _max_workspace_size = INVALID_WORKSPACE_SIZE;
    hipdnnPluginConstData_t _serialized_engine_config;
    flatbuffers::DetachedBuffer _engine_config_buffer;

    void set_engine(hipdnnBackendAttributeType_t attribute_type,
                    int64_t element_count,
                    const void* array_of_elements);

    void get_engine(hipdnnBackendAttributeType_t attribute_type,
                    int64_t requested_element_count,
                    int64_t* element_count,
                    void* array_of_elements) const;

    void get_max_workspace_size(hipdnnBackendAttributeType_t attribute_type,
                                int64_t requested_element_count,
                                int64_t* element_count,
                                void* array_of_elements) const;

    const hipdnnPluginConstData_t& get_serialized_engine_config();

public:
    static constexpr int64_t INVALID_WORKSPACE_SIZE = -1;

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

    static hipdnnBackendDescriptorType_t get_static_type();

    // Throws an exception if the descriptor is not finalized before calling these.
    std::shared_ptr<const Engine_descriptor> get_engine() const;
};

} // namespace hipdnn_backend
