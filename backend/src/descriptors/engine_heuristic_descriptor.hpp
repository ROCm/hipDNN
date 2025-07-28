// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <vector>

#include "backend_descriptor.hpp"

namespace hipdnn_backend
{

class Graph_descriptor;

class Engine_heuristic_descriptor : public hipdnnBackendDescriptorImpl<Engine_heuristic_descriptor>
{
private:
    std::shared_ptr<const Graph_descriptor> _graph;
    std::vector<int64_t> _engine_ids;
    bool _engine_ids_set = false;
    hipdnnBackendHeurMode_t _heuristic_mode = HIPDNN_HEUR_MODE_FALLBACK;
    bool _heuristic_mode_set = false;

    void set_graph(hipdnnBackendAttributeType_t attribute_type,
                   int64_t element_count,
                   const void* array_of_elements);

    void set_heuristic_mode(hipdnnBackendAttributeType_t attribute_type,
                            int64_t element_count,
                            const void* array_of_elements);

    void get_graph(hipdnnBackendAttributeType_t attribute_type,
                   int64_t requested_element_count,
                   int64_t* element_count,
                   void* array_of_elements) const;

    void get_engine_configs(hipdnnBackendAttributeType_t attribute_type,
                            int64_t requested_element_count,
                            int64_t* element_count,
                            void* array_of_elements) const;

    void get_heuristic_mode(hipdnnBackendAttributeType_t attribute_type,
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

    void set_engine_ids(const std::vector<int64_t>& engine_ids);

    // Throws an exception if the descriptor is not finalized.
    std::shared_ptr<const Graph_descriptor> get_graph() const;

    static hipdnnBackendDescriptorType_t get_static_type();
};

} // namespace hipdnn_backend
