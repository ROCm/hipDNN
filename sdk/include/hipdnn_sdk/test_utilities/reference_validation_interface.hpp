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

    virtual bool all_close(Migratable_memory_interface<T>& reference,
                           Migratable_memory_interface<T>& implementation)
        = 0;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk

// NOLINTEND(portability-template-virtual-member-function)
