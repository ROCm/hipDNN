// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <fcntl.h>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <regex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

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

        // pipe stderr to capture log output
        _old_stderr = dup(STDERR_FILENO);
        ASSERT_NE(_old_stderr, -1);
        ASSERT_EQ(pipe(_stderr_pipe.data()), 0);
        ASSERT_NE(dup2(_stderr_pipe[1], STDERR_FILENO), -1);
        ASSERT_EQ(close(_stderr_pipe[1]), 0);
        setenv("HIPDNN_LOG_LEVEL", "off", 1);
        unsetenv("HIPDNN_LOG_FILE");
    }

    void TearDown() override
    {
        hipdnn_backend::logging::cleanup();

        unsetenv("HIPDNN_LOG_LEVEL");
        unsetenv("HIPDNN_LOG_FILE");

        fflush(stderr);
        dup2(_old_stderr, STDERR_FILENO);
        close(_old_stderr);
        close(_stderr_pipe[0]);

        if(!_log_file.empty())
        {
            std::remove(_log_file.c_str());
        }
    }

    std::string get_stderr_content()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::string content;
        std::array<char, 4096> buffer;
        ssize_t bytes_read;

        fcntl(_stderr_pipe[0], F_SETFL, O_NONBLOCK);

        while((bytes_read = read(_stderr_pipe[0], buffer.data(), buffer.size() - 1)) > 0)
        {
            buffer[static_cast<size_t>(bytes_read)] = '\0';
            content += buffer.data();
        }
        return content;
    }

    void verify_stderr_contains(const std::string& expected_content)
    {
        std::string log_content = get_stderr_content();
        EXPECT_NE(log_content.find(expected_content), std::string::npos)
            << "Expected to find: \"" << expected_content << "\" in stderr."
            << "\nActual stderr content:\n"
            << log_content;
    }

    void verify_stderr_not_contains(const std::string& unexpected_content)
    {
        std::string log_content = get_stderr_content();
        EXPECT_EQ(log_content.find(unexpected_content), std::string::npos)
            << "Expected NOT to find: \"" << unexpected_content << "\" in stderr."
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
    EXPECT_TRUE(log_content.empty()) << "Expected stderr to be empty, but it contained:\n"
                                     << log_content;
}

TEST_F(Backend_logging_test, MacrosRespectLogLevelInfo)
{
    setenv("HIPDNN_LOG_LEVEL", "info", 1);

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
    setenv("HIPDNN_LOG_LEVEL", "warn", 1);

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
    setenv("HIPDNN_LOG_LEVEL", "error", 1);

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
    setenv("HIPDNN_LOG_LEVEL", "off", 1);
    HIPDNN_LOG_INFO("This should not appear");

    verify_stderr_not_contains("This should not appear");

    hipdnn_backend::logging::cleanup();

    setenv("HIPDNN_LOG_LEVEL", "info", 1);
    HIPDNN_LOG_INFO("This should appear after reinitialization");

    verify_stderr_contains("This should appear after reinitialization");
}

TEST_F(Backend_logging_test, LogPatternFormatIsCorrectOnStderr)
{
    setenv("HIPDNN_LOG_LEVEL", "info", 1);

    HIPDNN_LOG_INFO("Pattern format test message");

    std::string log_content = get_stderr_content();

    // [timestamp format] [thread id] [log level] [hipdnn_backend] message
    std::regex pattern_regex(
        R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[tid \d+\] \[info\] \[hipdnn_backend\] Pattern format test message)");

    EXPECT_TRUE(std::regex_search(log_content, pattern_regex))
        << "Expected log format pattern not found. Stderr content:\n"
        << log_content;
}

TEST_F(Backend_logging_test, MultipleMessagesAreLoggedToStderr)
{
    setenv("HIPDNN_LOG_LEVEL", "info", 1);

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

    EXPECT_TRUE(pos1 < pos2 && pos2 < pos3) << "Messages not logged in expected order to stderr";
}

TEST_F(Backend_logging_test, LogFileCanBeSpecifiedByEnvVar)
{
    _log_file = "custom_backend_test.log";
    setenv("HIPDNN_LOG_FILE", _log_file.c_str(), 1);
    setenv("HIPDNN_LOG_LEVEL", "info", 1);

    HIPDNN_LOG_INFO("Logging to custom file");

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    hipdnn_backend::logging::cleanup();

    std::string log_content;
    std::ifstream log_file_stream(_log_file);
    ASSERT_TRUE(log_file_stream.is_open()) << "Log file was not created: " << _log_file;

    log_content.assign((std::istreambuf_iterator<char>(log_file_stream)),
                       std::istreambuf_iterator<char>());
    log_file_stream.close();

    EXPECT_NE(log_content.find("Logging to custom file"), std::string::npos)
        << "Expected to find message in log file " << _log_file << "\nActual log content:\n"
        << log_content;

    verify_stderr_not_contains("Logging to custom file");
}
