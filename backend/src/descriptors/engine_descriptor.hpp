// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "backend_descriptor.hpp"

namespace hipdnn_backend
{

class Graph_descriptor;

class Engine_descriptor : public hipdnnBackendDescriptor
{
private:
    const Graph_descriptor* _graph = nullptr;
    int64_t _engine_id;
    bool _engine_id_set = false;

    void set_graph(hipdnnBackendAttributeType_t attribute_type,
                   int64_t element_count,
                   const void* array_of_elements);

    void get_graph(hipdnnBackendAttributeType_t attribute_type,
                   int64_t requested_element_count,
                   int64_t* element_count,
                   void* array_of_elements);

    void set_global_id(hipdnnBackendAttributeType_t attribute_type,
                       int64_t element_count,
                       const void* array_of_elements);

    void get_global_id(hipdnnBackendAttributeType_t attribute_type,
                       int64_t requested_element_count,
                       int64_t* element_count,
                       void* array_of_elements) const;

public:
    Engine_descriptor();
    ~Engine_descriptor() override = default;

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
