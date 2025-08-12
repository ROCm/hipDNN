// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "gtest/internal/gtest-port.h"
#include <fcntl.h>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_sdk/utilities/platform_utils.hpp>
#include <logging/logging.hpp>

class Backend_logging_test : public ::testing::Test
{
public:
    std::string _log_file;
    std::array<int, 2> _stderr_pipe;
    int _old_stderr;

    void SetUp() override
    {
        hipdnn_backend::logging::cleanup();

        testing::internal::CaptureStderr();

        hipdnn_sdk::utilities::set_env("HIPDNN_LOG_LEVEL", "off");
        hipdnn_sdk::utilities::unset_env("HIPDNN_LOG_FILE");
    }

    void TearDown() override
    {
        hipdnn_backend::logging::cleanup();

        hipdnn_sdk::utilities::unset_env("HIPDNN_LOG_LEVEL");
        hipdnn_sdk::utilities::unset_env("HIPDNN_LOG_FILE");

        if(!_log_file.empty())
        {
            std::remove(_log_file.c_str());
        }
    }

    static std::string get_stderr_content()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        return testing::internal::GetCapturedStderr();
    }

    static void verify_stderr_contains(const std::string& expected_content)
    {
        std::string log_content = get_stderr_content();
        EXPECT_NE(log_content.find(expected_content), std::string::npos)
            << std::string("Expected to find: \"") << expected_content << "\" in stderr."
            << "\nActual stderr content:\n"
            << log_content;
    }

    static void verify_stderr_not_contains(const std::string& unexpected_content)
    {
        std::string log_content = get_stderr_content();
        EXPECT_EQ(log_content.find(unexpected_content), std::string::npos)
            << std::string("Expected NOT to find: \"") << unexpected_content << "\" in stderr."
            << "\nActual stderr content:\n"
            << log_content;
    }
};

TEST_F(Backend_logging_test, MacrosDontLogWhenOff)
{
    HIPDNN_LOG_INFO("Initializing with info message");
    HIPDNN_LOG_WARN("Initializing with warn message");
    HIPDNN_LOG_ERROR("Initializing with error message");

    std::string log_content = get_stderr_content();
    EXPECT_TRUE(log_content.empty())
        << std::string("Expected stderr to be empty, but it contained:\n") << log_content;
}

TEST_F(Backend_logging_test, MacrosRespectLogLevelInfo)
{
    hipdnn_sdk::utilities::set_env("HIPDNN_LOG_LEVEL", "info");

    HIPDNN_LOG_INFO("Info test message");
    HIPDNN_LOG_WARN("Warn test message");
    HIPDNN_LOG_ERROR("Error test message");

    std::string log_content = get_stderr_content();
    EXPECT_NE(log_content.find("Info test message"), std::string::npos);
    EXPECT_NE(log_content.find("Warn test message"), std::string::npos);
    EXPECT_NE(log_content.find("Error test message"), std::string::npos);
}

TEST_F(Backend_logging_test, MacrosRespectLogLevelWarn)
{
    hipdnn_sdk::utilities::set_env("HIPDNN_LOG_LEVEL", "warn");

    HIPDNN_LOG_INFO("Info should not appear");
    HIPDNN_LOG_WARN("Warn should appear");
    HIPDNN_LOG_ERROR("Error should appear");

    std::string log_content = get_stderr_content();
    EXPECT_EQ(log_content.find("Info should not appear"), std::string::npos);
    EXPECT_NE(log_content.find("Warn should appear"), std::string::npos);
    EXPECT_NE(log_content.find("Error should appear"), std::string::npos);
}

TEST_F(Backend_logging_test, MacrosRespectLogLevelError)
{
    hipdnn_sdk::utilities::set_env("HIPDNN_LOG_LEVEL", "error");

    HIPDNN_LOG_INFO("Info should not appear");
    HIPDNN_LOG_WARN("Warn should not appear");
    HIPDNN_LOG_ERROR("Error should appear");

    std::string log_content = get_stderr_content();
    EXPECT_EQ(log_content.find("Info should not appear"), std::string::npos);
    EXPECT_EQ(log_content.find("Warn should not appear"), std::string::npos);
    EXPECT_NE(log_content.find("Error should appear"), std::string::npos);
}

TEST_F(Backend_logging_test, LoggingCanBeReinitialized)
{
    hipdnn_sdk::utilities::set_env("HIPDNN_LOG_LEVEL", "off");
    HIPDNN_LOG_INFO("This should not appear");

    hipdnn_backend::logging::cleanup();

    hipdnn_sdk::utilities::set_env("HIPDNN_LOG_LEVEL", "info");
    HIPDNN_LOG_INFO("This should appear after reinitialization");

    verify_stderr_contains("This should appear after reinitialization");
}

TEST_F(Backend_logging_test, LogPatternFormatIsCorrectOnStderr)
{
    hipdnn_sdk::utilities::set_env("HIPDNN_LOG_LEVEL", "info");

    HIPDNN_LOG_INFO("Pattern format test message");

    std::string log_content = get_stderr_content();

    // [timestamp format] [thread id] [log level] [hipdnn_backend] message
    std::regex pattern_regex(
        R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[tid \d+\] \[info\] \[hipdnn_backend\] Pattern format test message)");

    EXPECT_TRUE(std::regex_search(log_content, pattern_regex))
        << std::string("Expected log format pattern not found. Stderr content:\n") << log_content;
}

TEST_F(Backend_logging_test, MultipleMessagesAreLoggedToStderr)
{
    hipdnn_sdk::utilities::set_env("HIPDNN_LOG_LEVEL", "info");

    HIPDNN_LOG_INFO("First backend message");
    HIPDNN_LOG_INFO("Second backend message");
    HIPDNN_LOG_INFO("Third backend message");

    std::string log_content = get_stderr_content();
    EXPECT_NE(log_content.find("First backend message"), std::string::npos);
    EXPECT_NE(log_content.find("Second backend message"), std::string::npos);
    EXPECT_NE(log_content.find("Third backend message"), std::string::npos);

    // Verify expected order
    size_t pos1 = log_content.find("First backend message");
    size_t pos2 = log_content.find("Second backend message");
    size_t pos3 = log_content.find("Third backend message");

    EXPECT_TRUE(pos1 < pos2 && pos2 < pos3)
        << std::string("Messages not logged in expected order to stderr");
}

TEST_F(Backend_logging_test, LogFileCanBeSpecifiedByEnvVar)
{
    _log_file = "custom_backend_test.log";
    hipdnn_sdk::utilities::set_env("HIPDNN_LOG_FILE", _log_file.c_str());
    hipdnn_sdk::utilities::set_env("HIPDNN_LOG_LEVEL", "info");

    HIPDNN_LOG_INFO("Logging to custom file");

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    hipdnn_backend::logging::cleanup();

    std::string log_content;
    std::ifstream log_file_stream(_log_file);
    ASSERT_TRUE(log_file_stream.is_open())
        << std::string("Log file was not created: ") << _log_file;

    log_content.assign((std::istreambuf_iterator<char>(log_file_stream)),
                       std::istreambuf_iterator<char>());
    log_file_stream.close();

    EXPECT_NE(log_content.find("Logging to custom file"), std::string::npos)
        << std::string("Expected to find message in log file ") << _log_file
        << "\nActual log content:\n"
        << log_content;

    verify_stderr_not_contains("Logging to custom file");
}
