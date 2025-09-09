// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <string>

namespace hipdnn_sdk::test_utilities
{

class EnvironmentVariableGuard
{
public:
    explicit EnvironmentVariableGuard(const std::string& varName)
        : _varName(varName)
    {
        _originalValue = hipdnn_sdk::utilities::getEnv(varName.c_str(), "");
        _hadOriginalValue = !hipdnn_sdk::utilities::getEnv(varName.c_str()).empty();
    }

    ~EnvironmentVariableGuard()
    {
        if(_hadOriginalValue)
        {
            hipdnn_sdk::utilities::setEnv(_varName.c_str(), _originalValue.c_str());
        }
        else
        {
            hipdnn_sdk::utilities::unsetEnv(_varName.c_str());
        }
    }

    EnvironmentVariableGuard(const EnvironmentVariableGuard&) = delete;
    EnvironmentVariableGuard& operator=(const EnvironmentVariableGuard&) = delete;

    EnvironmentVariableGuard(EnvironmentVariableGuard&&) = default;
    EnvironmentVariableGuard& operator=(EnvironmentVariableGuard&&) = default;

private:
    std::string _varName;
    std::string _originalValue;
    bool _hadOriginalValue;
};

} // namespace hipdnn_sdk::test_utilities
