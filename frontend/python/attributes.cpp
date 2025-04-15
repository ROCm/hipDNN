// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <hipdnn_frontend/attributes/tensor_attributes.hpp>

namespace py = pybind11;

namespace hipdnn_frontend
{

namespace pythonapi
{

void init_attributes(py::module_& m)
{
    py::class_<graph::Tensor_attributes, std::shared_ptr<graph::Tensor_attributes>>(m, "tensor")
        .def(py::init<>())
        .def("get_uid", &graph::Tensor_attributes::get_uid)
        .def("get_name", &graph::Tensor_attributes::get_name)
        .def("get_data_type", &graph::Tensor_attributes::get_data_type)
        .def("get_stride", &graph::Tensor_attributes::get_stride)
        .def("get_dim", &graph::Tensor_attributes::get_dim)
        .def("get_volume", &graph::Tensor_attributes::get_volume)
        .def("get_is_virtual", &graph::Tensor_attributes::get_is_virtual)
        .def("has_uid", &graph::Tensor_attributes::has_uid)
        .def("set_uid", &graph::Tensor_attributes::set_uid)
        .def("set_name", &graph::Tensor_attributes::set_name)
        .def("set_data_type", &graph::Tensor_attributes::set_data_type)
        .def("_set_data_type", &graph::Tensor_attributes::set_data_type)
        .def("set_stride", &graph::Tensor_attributes::set_stride)
        .def("set_dim", &graph::Tensor_attributes::set_dim)
        .def("set_is_virtual", &graph::Tensor_attributes::set_is_virtual)
        .def("set_output", &graph::Tensor_attributes::set_output)
        .def("clear_uid", &graph::Tensor_attributes::clear_uid);
}

} // namespace pythonapi

} // namespace hipdnn_frontend