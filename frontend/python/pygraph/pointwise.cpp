// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <utility>

#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "pygraph.hpp"
#include <hipdnn_frontend/attributes/pointwise_attributes.hpp>

namespace py = pybind11;
using namespace pybind11::literals;

namespace hipdnn_frontend::python_api
{

std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>
    PyGraph::relu(std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& input,
                  std::optional<float> const& negative_slope,
                  std::optional<float> const& lower_clip,
                  std::optional<float> const& upper_clip,
                  hipdnn_frontend::DataType_t const& compute_data_type,
                  std::string const& name)
{
    auto attributes = hipdnn_frontend::graph::Pointwise_attributes()
                          // TODO .set_compute_data_type(compute_data_type)
                          .set_mode(hipdnn_frontend::PointwiseMode_t::RELU_FWD)
                          .set_name(name);

    if(negative_slope.has_value())
    {
        attributes.set_relu_lower_clip_slope(negative_slope.value());
    }

    if(lower_clip.has_value())
    {
        attributes.set_relu_lower_clip(lower_clip.value());
    }

    if(upper_clip.has_value())
    {
        attributes.set_relu_upper_clip(upper_clip.value());
    }

    auto OUT_0 = _graph->pointwise(input, attributes);
    return OUT_0;
}

void init_pygraph_pointwise_submodule(py::class_<PyGraph>& m)
{
    m.def("relu",
          &PyGraph::relu,
          py::arg("input"),
          py::arg_v("negative_slope", py::none()),
          py::arg_v("lower_clip", py::none()),
          py::arg_v("upper_clip", py::none()),
          py::arg_v("compute_data_type", hipdnn_frontend::DataType_t::NOT_SET),
          py::arg_v("name", ""),
          R"pbdoc(
      Apply the Rectified Linear Unit (ReLU) activation function to the input.

      Args:
          input (hipdnn_tensor): The input tensor.
          negative_slope (Optional[float]): Sets the lower clip slope value for ReLU.
          lower_clip (Optional[float]): Sets the lower clip value for ReLU.
          upper_clip (Optional[float]): Sets the upper clip value for ReLU.
          compute_data_type (Optional[hipdnn.data_type]): The data type for computation. Default is NOT_SET.
          name (Optional[str]): A name for the operation to be performed.

      Returns:
          hipdnn_tensor: The result of the ReLU activation.
      )pbdoc");
}

} // namespace hipdnn_frontend::python_api