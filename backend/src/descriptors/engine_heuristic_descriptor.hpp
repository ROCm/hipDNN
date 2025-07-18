// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <vector>

#include "backend_descriptor.hpp"

namespace hipdnn_backend
{

class Graph_descriptor;

class Engine_heuristic_descriptor : public hipdnnBackendDescriptor
{
private:
    const Graph_descriptor* _graph = nullptr;
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
                   void* array_of_elements);

    void get_engine_configs(hipdnnBackendAttributeType_t attribute_type,
                            int64_t requested_element_count,
                            int64_t* element_count,
                            void* array_of_elements);

    void get_heuristic_mode(hipdnnBackendAttributeType_t attribute_type,
                            int64_t requested_element_count,
                            int64_t* element_count,
                            void* array_of_elements);

public:
    Engine_heuristic_descriptor();
    ~Engine_heuristic_descriptor() override = default;

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

    void set_engine_ids(const std::vector<int64_t>& engine_ids);
};

} // namespace hipdnn_backend
