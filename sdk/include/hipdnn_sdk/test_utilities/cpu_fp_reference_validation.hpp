// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(__HIP_PLATFORM_AMD__)
// Need these for the half and bfloat16 types
#include <hipdnn_sdk/utilities/half_utils.hpp>
#include <hipdnn_sdk/utilities/hip_bfloat16_utils.hpp>
#endif

#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/test_utilities/reference_validation_interface.hpp>

namespace hipdnn_sdk
{
namespace reference_test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class T>
class Cpu_fp_reference_validation : public Reference_validation_interface<T>
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

    bool all_close(Migratable_memory_interface<T>& reference,
                   Migratable_memory_interface<T>& implementation) override
    {
        if(reference.count() != implementation.count())
        {
            return false;
        }

        size_t element_count = reference.count();

        const T* ref_data = reference.host_data();
        const T* impl_data = implementation.host_data();

        for(size_t i = 0; i < element_count; ++i)
        {
            T ref_value = ref_data[i];
            T impl_value = impl_data[i];

            T abs_diff = std::fabs(impl_value - ref_value);
            T threshold = _absolute_tolerance + _relative_tolerance * std::fabs(ref_value);

            if(abs_diff > threshold)
            {
                HIPDNN_LOG_ERROR("Validation failed at index {}: reference value = {}, "
                                 "implementation value = {}, "
                                 "absolute difference = {}, threshold = {} (atol={}, rtol={})",
                                 i,
                                 ref_value,
                                 impl_value,
                                 abs_diff,
                                 threshold,
                                 _absolute_tolerance,
                                 _relative_tolerance);
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
