// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/test_utilities/reference_validation_interface.hpp>

namespace hipdnn_sdk
{
namespace reference_test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class T>
class Cpu_fp_reference_validation : public Reference_validation_interface
{
public:
    Cpu_fp_reference_validation(T absolute_tolerance = std::numeric_limits<T>::epsilon(),
                                T relative_tolerance = std::numeric_limits<T>::epsilon())
        : _absolute_tolerance(absolute_tolerance)
        , _relative_tolerance(relative_tolerance)
    {
        if(absolute_tolerance < static_cast<T>(0.0f) || relative_tolerance < static_cast<T>(0.0f))
        {
            throw std::invalid_argument("Tolerances must be non-negative");
        }
    }

    ~Cpu_fp_reference_validation() override = default;

    bool compare_buffers(const Migratable_memory& reference,
                         const Migratable_memory& implementation) override
    {
        if(reference.count() != implementation.count())
        {
            return false;
        }

        size_t element_count = reference.count();
        const T* ref_data = reference.host_data<T>();
        const T* impl_data = implementation.host_data<T>();

        for(size_t i = 0; i < element_count; ++i)
        {

            T ref_value = ref_data[i];
            T impl_value = impl_data[i];

            T abs_diff = std::fabs(ref_value - impl_value);

            if(abs_diff > _absolute_tolerance
               && abs_diff
                      > _relative_tolerance * std::max(std::fabs(ref_value), std::fabs(impl_value)))
            {
                HIPDNN_LOG_ERROR("Validation failed at index {}: reference value = {}, "
                                 "implementation value = {}, "
                                 "absolute difference = {}, relative difference = {}",
                                 i,
                                 ref_value,
                                 impl_value,
                                 abs_diff,
                                 _relative_tolerance
                                     * std::max(std::fabs(ref_value), std::fabs(impl_value)));
                return false;
            }
        }

        return true;
    }

private:
    // Tolerances for comparison
    T _absolute_tolerance;
    T _relative_tolerance;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk
