// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "Error.hpp"
#include "logging/Logging.hpp"
#include <hipdnn_sdk/utilities/StringUtil.hpp>

// NOLINTNEXTLINE
thread_local char hipdnn_backend::LastErrorManager::s_lastError[HIPDNN_MAX_ERROR_STRING_SIZE] = "";

hipdnnStatus_t hipdnn_backend::LastErrorManager::setLastError(hipdnnStatus_t status,
                                                              const char* message)
{
    if(status == HIPDNN_STATUS_SUCCESS)
    {
        return status;
    }

    HIPDNN_LOG_ERROR(
        "Error occured in status:{} message:{}", hipdnnGetStatusString(status), message);

    hipdnn::sdk::utilities::copyMaxSizeWithNullTerminator(
        s_lastError, message, HIPDNN_MAX_ERROR_STRING_SIZE);

    return status;
}

hipdnnStatus_t hipdnn_backend::LastErrorManager::setLastError(hipdnnStatus_t status,
                                                              const std::string& message)
{
    return setLastError(status, message.c_str());
}

const char* hipdnn_backend::LastErrorManager::getLastError()
{
    return s_lastError;
}
