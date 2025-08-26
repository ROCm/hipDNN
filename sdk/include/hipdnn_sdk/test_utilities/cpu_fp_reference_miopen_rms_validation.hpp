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

// CPU validator that uses MIOpen RMS calculation for comparing tensor likes.
// Can be used to replicate MIOpen's RMS tolerance checks in unit tests.
// Note that this class does not use the absolute tolerance value, as MIOpen's
// RMS check is only relative tolerance based. We recommend using cpu_fp_reference_validation
// instead, but this class can be used to compare with MIOpen tolerance checks.
template <class T>
class Cpu_fp_reference_miopen_rms_validation : public Reference_validation_interface<T>
{
public:
    Cpu_fp_reference_miopen_rms_validation(T relative_tolerance = std::numeric_limits<T>::epsilon())
        : _relative_tolerance(static_cast<double>(relative_tolerance))
    {
        if(relative_tolerance < T{0})
        {
            throw std::invalid_argument("Tolerances must be non-negative");
        }
    }

    ~Cpu_fp_reference_miopen_rms_validation() override = default;

    bool all_close(Migratable_memory_interface<T>& reference,
                   Migratable_memory_interface<T>& implementation) override
    {
        if(reference.count() != implementation.count())
        {
            return false;
        }

        size_t element_count = reference.count();

        if(element_count == 0)
        {
            return true;
        }

        const T* ref_data = reference.host_data();
        const T* impl_data = implementation.host_data();

        double square_difference = 0.0;
        double max_ref_magnitude = 0.0;
        double max_impl_magnitude = 0.0;

        // Iterate through all elements to calculate square differences and find max magnitudes
        for(size_t i = 0; i < element_count; ++i)
        {
            auto ref_value = static_cast<double>(ref_data[i]);
            auto impl_value = static_cast<double>(impl_data[i]);

            // Accumulate square differences
            auto diff = ref_value - impl_value;
            square_difference += diff * diff;

            // Track maximum magnitudes
            max_ref_magnitude = std::max(max_ref_magnitude, std::fabs(ref_value));
            max_impl_magnitude = std::max(max_impl_magnitude, std::fabs(impl_value));
        }

        // Find the maximum magnitude between reference and implementation
        double max_magnitude
            = std::max({max_ref_magnitude, max_impl_magnitude, std::numeric_limits<double>::min()});

        double relative_rms_error
            = std::sqrt(square_difference)
              / (std::sqrt(static_cast<double>(element_count)) * max_magnitude);

        return relative_rms_error <= _relative_tolerance;
    }

private:
    // Tolerance for comparison
    double _relative_tolerance;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk
