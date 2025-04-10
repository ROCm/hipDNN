// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

namespace hipdnn_frontend
{
    namespace graph
    {
        enum class PointwiseMode_t
        {
            NOT_SET = 0,
            RELU    = 1,
        };

        enum class DataType_t
        {
            NOT_SET  = 0,
            FLOAT    = 1,
            HALF     = 2,
            BFLOAT16 = 3,
        };
    }
}