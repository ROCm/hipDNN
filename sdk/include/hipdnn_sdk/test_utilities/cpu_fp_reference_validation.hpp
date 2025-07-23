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
    Cpu_fp_reference_validation(T absolute_tolerance, T relative_tolerance)
        : _absolute_tolerance(absolute_tolerance), _relative_tolerance(relative_tolerance)
    {
        if (absolute_tolerance < 0 || relative_tolerance < 0) {
            throw std::invalid_argument("Tolerances must be non-negative");
        }
    }
    
    ~Cpu_fp_reference_validation() override = default;
    
    bool compare_buffers(const Migratable_memory& reference, const Migratable_memory& implementation) override
    {
        if (reference.count() != implementation.count()) {
            return false;
        }

        size_t element_count = reference.count();
        const T* ref_data = reference.host_data<T>();
        const T* impl_data = implementation.host_data<T>();

        for (size_t i = 0; i < element_count; ++i) {

            T ref_value = ref_data[i];
            T impl_value = impl_data[i];

            T abs_diff = std::fabs(ref_value - impl_value);

            // Based this on: https://realtimecollisiondetection.net/blog/?p=89
            if (abs_diff > _absolute_tolerance &&
                abs_diff > _relative_tolerance * std::max(std::fabs(ref_value), std::fabs(impl_value))) {
                
                return false;
            }
        }
        
        return true;
    }
    
private:
    // Tolerances for comparison
    T _absolute_tolerance = std::numeric_limits<T>::epsilon();
    T _relative_tolerance = std::numeric_limits<T>::epsilon();
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk
