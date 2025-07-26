// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "descriptors/backend_descriptor.hpp"

namespace test_descriptor_utils
{

template <typename T>
hipdnnBackendDescriptor* create_descriptor_ptr()
{
    return hipdnnBackendDescriptor::pack_descriptor(std::make_shared<T>());
}

template <typename T>
std::unique_ptr<hipdnnBackendDescriptor> create_descriptor()
{
    return std::unique_ptr<hipdnnBackendDescriptor>(create_descriptor_ptr<T>());
}
}
