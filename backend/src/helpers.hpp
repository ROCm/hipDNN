// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "error.hpp"
#include "hipdnn_exception.hpp"
#include "hipdnn_status.h"

#include <iostream>

namespace hipdnn_backend
{

template <class F>
hipdnnStatus_t try_catch(F f)
{
    try
    {
        f();
    }
    catch(const Hipdnn_exception& ex)
    {
        return set_last_error(ex.get_status(), ex.what());
    }
    catch(const std::exception& ex)
    {
        return set_last_error(HIPDNN_STATUS_INTERNAL_ERROR, ex.what());
    }
    catch(...)
    {
        return set_last_error(HIPDNN_STATUS_INTERNAL_ERROR, "Unknown exception occured");
    }
    return HIPDNN_STATUS_SUCCESS;
}

// TODO - remove this version once everything is swapped
template <class F>
hipdnnStatus_t try_catch_with_status(F f)
{
    hipdnnStatus_t status = HIPDNN_STATUS_SUCCESS;
    try
    {
        status = f();
    }
    catch(const Hipdnn_exception& ex)
    {
        return set_last_error(ex.get_status(), ex.what());
    }
    catch(const std::exception& ex)
    {
        return set_last_error(HIPDNN_STATUS_INTERNAL_ERROR, ex.what());
    }
    catch(...)
    {
        return set_last_error(HIPDNN_STATUS_INTERNAL_ERROR, "Unknown exception occured");
    }
    return status;
}
} // namespace hipdnn_backend