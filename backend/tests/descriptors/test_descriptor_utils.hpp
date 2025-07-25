// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "descriptors/backend_descriptor.hpp"

namespace test_descriptor_utils
{

template <typename T>
hipdnnBackendDescriptor* create_descriptor_ptr()
{
    auto desc_wrapper = new hipdnnBackendDescriptor();
    desc_wrapper->private_descriptor = std::make_shared<T>();

    return desc_wrapper;
}

template <typename T>
std::unique_ptr<hipdnnBackendDescriptor> create_descriptor()
{
    return std::unique_ptr<hipdnnBackendDescriptor>(create_descriptor_ptr<T>());
}
}
