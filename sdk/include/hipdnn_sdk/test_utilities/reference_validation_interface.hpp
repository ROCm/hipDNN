// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// NOLINTBEGIN(portability-template-virtual-member-function)

#include <hipdnn_sdk/utilities/migratable_memory.hpp>
#include <type_traits>

namespace hipdnn_sdk
{
namespace reference_test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class T>
class Reference_validation_interface
{
public:
    virtual ~Reference_validation_interface() = default;

    virtual bool compare_buffers(const Migratable_memory<T>& reference,
                                 const Migratable_memory<T>& implementation,
                                 hipStream_t stream = nullptr)
        = 0;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk

// NOLINTEND(portability-template-virtual-member-function)