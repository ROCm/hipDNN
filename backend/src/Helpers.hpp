// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Error.hpp"
#include "HipdnnException.hpp"
#include "HipdnnStatus.h"

#include <iostream>

namespace hipdnn_backend
{

template <class F>
hipdnnStatus_t tryCatch(F f)
{
    try
    {
        f();
    }
    catch(const HipdnnException& ex)
    {
        return LastErrorManager::setLastError(ex.getStatus(), ex.what());
    }
    catch(const std::exception& ex)
    {
        return LastErrorManager::setLastError(HIPDNN_STATUS_INTERNAL_ERROR, ex.what());
    }
    catch(...)
    {
        return LastErrorManager::setLastError(HIPDNN_STATUS_INTERNAL_ERROR,
                                              "Unknown exception occured");
    }
    return HIPDNN_STATUS_SUCCESS;
}
} // namespace hipdnn_backend
