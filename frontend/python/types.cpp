// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "pybind11/pybind11.h"

#include <hipdnn_frontend/types.hpp>

namespace py = pybind11;

namespace hipdnn_frontend
{

namespace python_api
{

void init_types(py::module_& m)
{
    py::enum_<DataType_t>(m, "data_type")
        .value("NOT_SET", DataType_t::NOT_SET)
        .value("FLOAT", DataType_t::FLOAT)
        .value("HALF", DataType_t::HALF)
        .value("BFLOAT16", DataType_t::BFLOAT16);
}

} // namespace python_api

} // namespace hipdnn_frontend