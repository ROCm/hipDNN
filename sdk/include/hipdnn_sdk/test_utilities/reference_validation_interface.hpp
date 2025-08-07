// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/utilities/migratable_memory.hpp>
#include <type_traits>

namespace hipdnn_sdk
{
namespace reference_test_utilities
{

using namespace hipdnn_sdk::utilities;

class Reference_validation_interface
{
public:
    virtual ~Reference_validation_interface() = default;

    virtual bool compare_buffers(const Migratable_memory& reference,
                                 const Migratable_memory& implementation)
        = 0;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk
