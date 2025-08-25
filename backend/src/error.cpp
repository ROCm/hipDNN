// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "error.hpp"
#include "logging/logging.hpp"
#include <hipdnn_sdk/utilities/string_util.hpp>

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char hipdnn_backend::LastErrorManager::lastError[HIPDNN_MAX_ERROR_STRING_SIZE] = "";

hipdnnStatus_t hipdnn_backend::LastErrorManager::setLastError(hipdnnStatus_t status,
                                                                  const char* message)
{
    if(status == HIPDNN_STATUS_SUCCESS)
    {
        return status;
    }

    HIPDNN_LOG_ERROR(
        "Error occured in status:{} message:{}", hipdnnGetStatusString(status), message);

    hipdnn::sdk::utilities::copy_max_size_with_null_terminator(
        lastError, message, HIPDNN_MAX_ERROR_STRING_SIZE);

    return status;
}

hipdnnStatus_t hipdnn_backend::LastErrorManager::setLastError(hipdnnStatus_t status,
                                                                  const std::string& message)
{
    return setLastError(status, message.c_str());
}

const char* hipdnn_backend::LastErrorManager::getLastError()
{
    return lastError;
}
