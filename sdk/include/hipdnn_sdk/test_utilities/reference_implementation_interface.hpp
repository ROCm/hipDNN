// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <cstdint>
#include <hipdnn_sdk/utilities/migratable_memory.hpp>
#include <hipdnn_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>

namespace hipdnn_sdk {
namespace reference_test_utilities {

using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

template<class T>
class Reference_implementation_interface {
public:
    virtual ~Reference_implementation_interface() = default;
    
    virtual void execute(const std::map<int64_t, Migratable_memory<T>>& buffers, const TensorAttributesT& tensor_attributes, const BatchnormInferenceAttributesT& batchnorm_attributes) = 0;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk