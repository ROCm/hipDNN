// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include "backend_descriptor.hpp"
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_backend
{

class Variant_descriptor : public hipdnnBackendDescriptorImpl<Variant_descriptor>
{
private:
    std::vector<const void*> _data_pointers;
    std::vector<int64_t> _unique_ids;
    void* _workspace = nullptr;

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

    // throws if the variant descriptor is not finalized
    void* get_workspace() const;
    const std::vector<const void*>& get_data_pointers() const;
    const std::vector<int64_t>& get_tensor_ids() const;

    static hipdnnBackendDescriptorType_t get_static_type();
};
}
