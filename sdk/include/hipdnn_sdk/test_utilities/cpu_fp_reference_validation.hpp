// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/reference_validation_interface.hpp>

namespace hipdnn_sdk {
namespace reference_test_utilities {

using namespace hipdnn_sdk::utilities;

template<class T>
class Cpu_fp_reference_validation : public Reference_validation_interface<T> {
public:
    
    Cpu_fp_reference_validation() = default;
    Cpu_fp_reference_validation(double absolute_tolerance, double relative_tolerance)
        : _absolute_tolerance(absolute_tolerance), _relative_tolerance(relative_tolerance)
    {
        if (absolute_tolerance < 0 || relative_tolerance < 0) {
            throw std::invalid_argument("Tolerances must be non-negative");
        }
    }
    
    ~Cpu_fp_reference_validation() override = default;
    
    bool compare_buffers(const Migratable_memory<T>& reference, const Migratable_memory<T>& implementation) override
    {
        if (reference.size() != implementation.size()) {
            return false;
        }

        size_t element_count = reference.size();
        const T* ref_data = reference.host_data();
        const T* impl_data = implementation.host_data();

        for (size_t i = 0; i < element_count; ++i) {

            // Convert to double for comparison
            // This is necessary for types like hip_bfloat16 and half which may not support direct
            // arithmetic operations in a straightforward way.
            double ref_value = ref_data[i];
            double impl_value = impl_data[i];

            double abs_diff = std::abs(ref_value - impl_value);

            // Based this on: https://realtimecollisiondetection.net/blog/?p=89
            if (abs_diff > _absolute_tolerance &&
                abs_diff > _relative_tolerance * std::max(std::abs(ref_value), std::abs(impl_value))) {
                
                return false;
            }
        }
        
        return true;
    }
    
private:
    // Tolerances for comparison
    double _absolute_tolerance = std::numeric_limits<double>::epsilon();
    double _relative_tolerance = std::numeric_limits<double>::epsilon();
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk
