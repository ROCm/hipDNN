// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "miopen_tensor.hpp"
#include "miopen_utils.hpp"

namespace miopen_legacy_plugin
{

Miopen_tensor::Miopen_tensor(const hipdnn_sdk::data_objects::TensorAttributes& tensor)
    : _uid(tensor.uid())
    , _descriptor(nullptr)
{
    THROW_ON_MIOPEN_FAILURE(miopenCreateTensorDescriptor(&_descriptor));

    std::vector<int> dims(tensor.dims()->begin(), tensor.dims()->end());
    std::vector<int> strides(tensor.strides()->begin(), tensor.strides()->end());
    THROW_ON_MIOPEN_FAILURE(miopenSetTensorDescriptor(
        _descriptor,
        miopen_utils::tensor_data_type_to_miopen_data_type(tensor.data_type()),
        static_cast<int>(dims.size()),
        reinterpret_cast<int*>(dims.data()),
        reinterpret_cast<int*>(strides.data())));
}

Miopen_tensor::~Miopen_tensor()
{
    if(_descriptor != nullptr)
    {
        LOG_ON_MIOPEN_FAILURE(miopenDestroyTensorDescriptor(_descriptor));
    }
}

int64_t Miopen_tensor::uid() const
{
    return _uid;
}

miopenTensorDescriptor_t Miopen_tensor::tensor_descriptor() const
{
    return _descriptor;
}

}
