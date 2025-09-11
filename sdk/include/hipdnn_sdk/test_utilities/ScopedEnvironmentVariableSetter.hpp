// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdlib>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <string>

namespace hipdnn_sdk::test_utilities
{

class ScopedEnvironmentVariableSetter
{
public:
    explicit ScopedEnvironmentVariableSetter(const std::string& varName)
        : _varName(varName)
    {
        auto originalValue = std::getenv(varName.c_str());
        _hadOriginalValue = (originalValue != nullptr);
        _originalValue = _hadOriginalValue ? originalValue : "";
    }

    ~ScopedEnvironmentVariableSetter()
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

    ScopedEnvironmentVariableSetter(const ScopedEnvironmentVariableSetter&) = delete;
    ScopedEnvironmentVariableSetter& operator=(const ScopedEnvironmentVariableSetter&) = delete;

    ScopedEnvironmentVariableSetter(ScopedEnvironmentVariableSetter&&) = default;
    ScopedEnvironmentVariableSetter& operator=(ScopedEnvironmentVariableSetter&&) = default;

private:
    std::string _varName;
    std::string _originalValue;
    bool _hadOriginalValue;
};

} // namespace hipdnn_sdk::test_utilities
