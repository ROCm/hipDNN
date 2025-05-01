// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <vector>

#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "pygraph.hpp"
#include <hipdnn_frontend/attributes/batchnorm_backward_attributes.hpp>
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>

namespace py = pybind11;
using namespace pybind11::literals;

namespace hipdnn_frontend::python_api
{

std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes> PyGraph::batchnorm_inference(
    std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& x,
    std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& mean,
    std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& inv_variance,
    std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& scale,
    std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& bias,
    hipdnn_frontend::DataType_t const& compute_data_type,
    std::string const& name)
{
    auto attributes
        = hipdnn_frontend::graph::Batchnorm_inference_attributes()
              //.set_compute_data_type(compute_data_type) TODO missing set_compute_data_type
              .set_name(name);
    return _graph->batchnorm_inference(x, mean, inv_variance, scale, bias, attributes);
}

std::vector<std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>> PyGraph::batchnorm_backward(
    std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes> const& dy,
    std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes> const& x,
    std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes> const& scale,
    std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes> const& mean,
    std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes> const& inv_variance,
    std::vector<std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>>& peer_stats,
    hipdnn_frontend::DataType_t const& compute_data_type,
    std::string const& name)
{
    auto attributes = hipdnn_frontend::graph::Batchnorm_backward_attributes()
                          .set_saved_mean_and_inv_variance(mean, inv_variance)
                          .set_peer_stats(peer_stats)
                          //.set_compute_data_type(compute_data_type) TODO missing
                          .set_name(name);

    auto [DX, DScale, DBias] = _graph->batchnorm_backward(dy, x, scale, attributes);
    return {DX, DScale, DBias};
}

void init_pygraph_norm_submodule(py::class_<PyGraph>& m)
{
    m.def("batchnorm_inference",
          &PyGraph::batchnorm_inference,
          py::arg("input"),
          py::arg("mean"),
          py::arg("inv_variance"),
          py::arg("scale"),
          py::arg("bias"),
          py::arg_v("compute_data_type", hipdnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""))
        .def("batchnorm_backward",
             &PyGraph::batchnorm_backward,
             py::arg("grad"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("mean"),
             py::arg("inv_variance"),
             py::arg_v("peer_stats",
                       std::vector<std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>>()),
             py::arg_v("compute_data_type", hipdnn_frontend::DataType_t::NOT_SET),
             py::arg_v("name", ""));
}

} // namespace hipdnn_frontend::python_api