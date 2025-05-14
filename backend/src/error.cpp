// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "error.hpp"
#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/utilities/string_util.hpp>

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char hipdnn_backend::Last_error_manager::last_error[HIPDNN_MAX_ERROR_STRING_SIZE] = "";

hipdnnStatus_t hipdnn_backend::Last_error_manager::set_last_error(hipdnnStatus_t status,
                                                                  const char* message)
{
    if(status == HIPDNN_STATUS_SUCCESS)
    {
        return status;
    }

    HIPDNN_LOG_ERROR(
        "Error occured in status:{} message:{}", hipdnn_get_status_string(status), message);

    hipdnn::sdk::utilities::copy_max_size_with_null_terminator(
        last_error, message, HIPDNN_MAX_ERROR_STRING_SIZE);

    return status;
}

hipdnnStatus_t hipdnn_backend::Last_error_manager::set_last_error(hipdnnStatus_t status,
                                                                  const std::string& message)
{
    return set_last_error(status, message.c_str());
}

const char* hipdnn_backend::Last_error_manager::get_last_error()
{
    return last_error;
}