/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
*/
#pragma once

#include <memory>

#include "mock_hipdnn_backend.hpp"

void set_mock_hipdnn_backend(std::weak_ptr<Mock_hipdnn_backend> backend);
