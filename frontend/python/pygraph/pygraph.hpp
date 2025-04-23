// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <hipdnn_frontend/graph.hpp>

namespace py = pybind11;
using namespace pybind11::literals;

namespace hipdnn_frontend::python_api
{

// This class wraps Graph_t to provide a more pythonic interface.
class PyGraph
{
public:
    using Graph_t = std::shared_ptr<hipdnn_frontend::graph::Graph>;

    PyGraph(Graph_t graph)
        : _graph(graph){};

    PyGraph(std::string const&          name,
            hipdnn_frontend::DataType_t io_data_type,
            hipdnn_frontend::DataType_t intermediate_data_type,
            hipdnn_frontend::DataType_t compute_data_type)
        : _graph(std::make_shared<hipdnn_frontend::graph::Graph>())
    {
        _graph->set_name(name)
            .set_io_data_type(io_data_type)
            .set_intermediate_data_type(intermediate_data_type)
            .set_compute_data_type(compute_data_type);

        //if (handle_.has_value()) {
        //    handle = static_cast<hipdnnHandle_t>((void*)(handle_.value()));
        //} else {
        //    detail::create_handle(&handle);
        //    is_handle_owner = true;
        //}
    }

    ~PyGraph()
    {
        //if (is_handle_owner) {
        //    detail::destroy_handle(handle);
        //}
    }

    std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes> batchnorm_inference(
        std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& x,
        std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& mean,
        std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& inv_variance,
        std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& scale,
        std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& bias,
        hipdnn_frontend::DataType_t const&                          compute_data_type,
        std::string const&                                          name);

    std::vector<std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>> batchnorm_backward(
        std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes> const&        dy,
        std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes> const&        x,
        std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes> const&        scale,
        std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes> const&        mean,
        std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes> const&        inv_variance,
        std::vector<std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>>& peer_stats,
        hipdnn_frontend::DataType_t const&                                       compute_data_type,
        std::string const&                                                       name);

    std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>
        relu(std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& input,
             std::optional<float> const&                                 negative_slope,
             std::optional<float> const&                                 lower_clip,
             std::optional<float> const&                                 upper_clip,
             hipdnn_frontend::DataType_t const&                          compute_data_type,
             std::string const&                                          name);

    void validate();

    void build_operation_graph();

    std::vector<uint8_t> serialize() const;

    void deserialize(py::object const& pyobj);

    // Helpers for basic unit testing
    const std::string& get_name() const
    {
        return _graph->get_name();
    }
    DataType_t get_compute_data_type() const
    {
        return _graph->get_compute_data_type();
    }
    DataType_t get_intermediate_data_type() const
    {
        return _graph->get_intermediate_data_type();
    }
    DataType_t get_io_data_type() const
    {
        return _graph->get_io_data_type();
    }

private:
    Graph_t _graph;
    //hipdnnHandle_t _handle;
    //bool _is_handle_owner = false;
};

} // namespace hipdnn_frontend::python_api