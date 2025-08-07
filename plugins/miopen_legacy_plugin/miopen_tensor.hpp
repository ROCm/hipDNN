// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <miopen/miopen.h>

namespace miopen_legacy_plugin
{

class Miopen_tensor
{
public:
    Miopen_tensor(const hipdnn_sdk::data_objects::TensorAttributes& tensor);

    ~Miopen_tensor();

    int64_t uid() const;

    miopenTensorDescriptor_t tensor_descriptor() const;

private:
    int64_t _uid;
    miopenTensorDescriptor_t _descriptor;
};

}
