// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

typedef enum
{
    HIPDNN_SEV_FATAL = 0,
    HIPDNN_SEV_ERROR = 1,
    HIPDNN_SEV_WARNING = 2,
    HIPDNN_SEV_INFO = 3,
    // more?
} hipdnnSeverity_t;