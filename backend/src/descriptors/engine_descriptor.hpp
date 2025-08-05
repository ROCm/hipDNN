// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "backend_descriptor.hpp"

namespace hipdnn_backend
{

class Graph_descriptor;

namespace plugin
{
class Engine_details_wrapper;
}

class Engine_descriptor : public hipdnnBackendDescriptorImpl<Engine_descriptor>
{
private:
    std::shared_ptr<const Graph_descriptor> _graph;
    int64_t _engine_id;
    bool _engine_id_set = false;
    std::shared_ptr<const plugin::Engine_details_wrapper> _engine_details;

    void set_graph(hipdnnBackendAttributeType_t attribute_type,
                   int64_t element_count,
                   const void* array_of_elements);

    void get_graph(hipdnnBackendAttributeType_t attribute_type,
                   int64_t requested_element_count,
                   int64_t* element_count,
                   void* array_of_elements) const;

    void set_global_id(hipdnnBackendAttributeType_t attribute_type,
                       int64_t element_count,
                       const void* array_of_elements);

    void get_global_id(hipdnnBackendAttributeType_t attribute_type,
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

    // These getters throw an exception if the descriptor is not finalized.
    virtual std::shared_ptr<const Graph_descriptor> get_graph() const;
    virtual int64_t get_engine_id() const;

    static hipdnnBackendDescriptorType_t get_static_type();
};

} // namespace hipdnn_backend
