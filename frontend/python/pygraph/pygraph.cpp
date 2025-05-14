// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "pygraph.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

namespace hipdnn_frontend::python_api
{

void throw_if(bool const cond,
              hipdnn_frontend::error_code_t const error_code,
              std::string const& error_msg);

void init_pygraph_norm_submodule(py::class_<PyGraph>&);

void init_pygraph_pointwise_submodule(py::class_<PyGraph>&);

void PyGraph::validate()
{
    auto status = _graph->validate();
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

void PyGraph::build_operation_graph()
{
    // TODO hipdnn build_operation_graph to take a handle
    auto status = _graph->build_operation_graph();
    throw_if(status.is_bad(), status.get_code(), status.get_message());
}

std::vector<uint8_t> PyGraph::serialize() const
{
    // TODO can be simplified when frontend serialize() method is added
    std::vector<uint8_t> data;
    const uint8_t* raw_data = _graph->serialized_graph.data();
    size_t size = _graph->serialized_graph.size();
    data.insert(data.end(), raw_data, raw_data + size);
    return data;
}

void PyGraph::deserialize(py::object const& pyobj)
{
    std::vector<uint8_t> data = pyobj.cast<std::vector<uint8_t>>();
    // TODO - need deserialize() frontend graph method
}

void init_pygraph_submodule(py::module_& m)
{
    py::class_<PyGraph> pygraph_(m, "pygraph");
    pygraph_
        .def(py::init<std::string const&, DataType_t, DataType_t, DataType_t>(),
             py::arg_v("name", "test_graph"),
             py::arg_v("io_data_type", DataType_t::NOT_SET),
             py::arg_v("intermediate_data_type", DataType_t::NOT_SET),
             py::arg_v("compute_data_type", DataType_t::NOT_SET))
        .def("get_name", &PyGraph::get_name)
        .def("get_io_data_type", &PyGraph::get_io_data_type)
        .def("get_intermediate_data_type", &PyGraph::get_intermediate_data_type)
        .def("get_compute_data_type", &PyGraph::get_compute_data_type)
        .def("validate", &PyGraph::validate)
        .def("build_operation_graph", &PyGraph::build_operation_graph)
        .def("serialize", &PyGraph::serialize)
        .def("deserialize", &PyGraph::deserialize);

    init_pygraph_norm_submodule(pygraph_);
    init_pygraph_pointwise_submodule(pygraph_);
}

} // namespace hipdnn_frontend::python_api
