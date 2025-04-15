// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hipdnn_frontend
{

namespace pythonapi
{

void init_types(py::module_&);
void init_attributes(py::module_&);

PYBIND11_MODULE(hipdnn, m)
{

    init_types(m);
    init_attributes(m);

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

} // namespace pythonapi

} // namespace hipdnn_frontend