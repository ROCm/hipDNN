// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <hipdnn_frontend/error.hpp>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hipdnn_frontend
{

namespace python_api
{

void init_types(py::module_&);
void init_attributes(py::module_&);
void init_pygraph_submodule(py::module_&);

// Raise C++ exceptions corresponding to C++ FE error codes.
// Pybind automatically converts C++ exceptions to python exceptions.
void throw_if(bool const                          cond,
              hipdnn_frontend::error_code_t const error_code,
              std::string const&                  error_msg)
{
    if(cond == false)
        return;

    switch(error_code)
    {
    case hipdnn_frontend::error_code_t::OK:
        return;
    case hipdnn_frontend::error_code_t::ATTRIBUTE_NOT_SET:
        throw std::invalid_argument(error_msg);
    case hipdnn_frontend::error_code_t::INVALID_VALUE:
        throw std::runtime_error(error_msg);
    }
}

PYBIND11_MODULE(hipdnn, m)
{

    init_types(m);
    init_attributes(m);
    init_pygraph_submodule(m);

    m.doc() = R"pbdoc(
        Pybind11 hipDNN frontend plugin
        -----------------------

        .. currentmodule:: hipdnn

        .. autosummary::
           :toctree: _generate

    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "0.0.1";
#endif
}

} // namespace python_api

} // namespace hipdnn_frontend