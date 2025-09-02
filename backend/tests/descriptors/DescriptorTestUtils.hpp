// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "descriptors/BackendDescriptor.hpp"

namespace test_descriptor_utils
{

template <typename T>
HipdnnBackendDescriptor* createDescriptorPtr()
{
    return HipdnnBackendDescriptor::packDescriptor(std::make_shared<T>());
}

template <typename T>
std::unique_ptr<HipdnnBackendDescriptor> createDescriptor()
{
    return std::unique_ptr<HipdnnBackendDescriptor>(createDescriptorPtr<T>());
}
}
