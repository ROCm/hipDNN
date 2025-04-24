// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "error.hpp"
#include "hipdnn_status.h"

#include <iostream>

namespace hipdnn_backend
{

template <class F>
hipdnnStatus_t try_catch(F f, bool output = true)
{
    hipdnnStatus_t status = HIPDNN_STATUS_SUCCESS;

    try
    {
        status = f();
    }
    catch(const std::exception& ex)
    {
        if(output)
        {
            std::cerr << "HipDNN Error: " << ex.what() << "\n";
        }
        return set_last_error(HIPDNN_STATUS_INTERNAL_ERROR, ex.what());
    }
    catch(...)
    {
        return set_last_error(HIPDNN_STATUS_INTERNAL_ERROR, "Unknown exception occured");
    }
    return status;
}
}