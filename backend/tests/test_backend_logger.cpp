// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <regex>
#include <spdlog/spdlog.h>
#include <string>
#include <thread>

#include <hipdnn_sdk/logging/logger.hpp>

// This test suite exclusively verifies the initialization and functionality of the internal backend logging mechanism.
class Backend_logging_test : public ::testing::Test
{
public:
    std::string _log_file;

    void SetUp() override
    {
        _log_file = hipdnn::logging::generate_log_file_name();
        
        hipdnn::logging::cleanup_logging();
        
        setenv("HIPDNN_LOG_LEVEL", "off", 1);
        setenv("HIPDNN_LOG_FILE", _log_file.c_str(), 1);
    }

    void TearDown() override
    {
        hipdnn::logging::cleanup_logging();
        
        unsetenv("HIPDNN_LOG_LEVEL");
        unsetenv("HIPDNN_LOG_FILE");
        
        if (!_log_file.empty())
        {
            std::ifstream log_file_stream(_log_file);
            if (log_file_stream.is_open())
            {
                log_file_stream.close();
                std::remove(_log_file.c_str());
            }
        }
    }

    std::string get_log_content() const
    {
        // Await pending async log operations. The backend logging is completely asynchronous.
        if (hipdnn::logging::g_backend_logger)
        {
            hipdnn::logging::g_backend_logger->flush();
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        std::string log_content;
        std::ifstream log_file_stream(_log_file);
        if (log_file_stream.is_open())
        {
            log_content.assign((std::istreambuf_iterator<char>(log_file_stream)),
                               std::istreambuf_iterator<char>());
            log_file_stream.close();
        }
        return log_content;
    }

    void verify_log_contains(const std::string& expected_content) const
    {
        std::string log_content = get_log_content();
        EXPECT_NE(log_content.find(expected_content), std::string::npos)
            << "Expected to find: \"" << expected_content << "\" in log file " << _log_file
            << "\nActual log content:\n" << log_content;
    }

    void verify_log_not_contains(const std::string& unexpected_content) const
    {
        std::string log_content = get_log_content();
        EXPECT_EQ(log_content.find(unexpected_content), std::string::npos)
            << "Expected NOT to find: \"" << unexpected_content << "\" in log file " << _log_file
            << "\nActual log content:\n" << log_content;
    }
};

TEST_F(Backend_logging_test, MacrosLazilyInitializeLoggingWithDefaultSettings)
{
    HIPDNN_LOG_INFO("Initializing with info message");
    HIPDNN_LOG_WARN("Initializing with warn message");
    HIPDNN_LOG_ERROR("Initializing with error message");
    
    EXPECT_TRUE(hipdnn::logging::g_logging_initialized);
    EXPECT_NE(hipdnn::logging::g_backend_logger, nullptr);
    
    // With default log level "off", nothing should be logged
    verify_log_not_contains("Initializing with info message");
    verify_log_not_contains("Initializing with warn message");
    verify_log_not_contains("Initializing with error message");
}

TEST_F(Backend_logging_test, MacrosRespectLogLevelInfo)
{
    setenv("HIPDNN_LOG_LEVEL", "info", 1);
    
    HIPDNN_LOG_INFO("Info test message");
    HIPDNN_LOG_WARN("Warn test message");
    HIPDNN_LOG_ERROR("Error test message");
    
    verify_log_contains("Info test message");
    verify_log_contains("Warn test message");
    verify_log_contains("Error test message");
}

TEST_F(Backend_logging_test, MacrosRespectLogLevelWarn)
{
    setenv("HIPDNN_LOG_LEVEL", "warn", 1);
    
    HIPDNN_LOG_INFO("Info should not appear");
    HIPDNN_LOG_WARN("Warn should appear");
    HIPDNN_LOG_ERROR("Error should appear");
    
    verify_log_not_contains("Info should not appear");
    verify_log_contains("Warn should appear");
    verify_log_contains("Error should appear");
}

TEST_F(Backend_logging_test, MacrosRespectLogLevelError)
{
    setenv("HIPDNN_LOG_LEVEL", "error", 1);
    
    HIPDNN_LOG_INFO("Info should not appear");
    HIPDNN_LOG_WARN("Warn should not appear");
    HIPDNN_LOG_ERROR("Error should appear");
    
    verify_log_not_contains("Info should not appear");
    verify_log_not_contains("Warn should not appear");
    verify_log_contains("Error should appear");
}

TEST_F(Backend_logging_test, LogFileCanBeSpecifiedByEnvVar)
{
    std::string custom_log_file = "custom_backend_test.log";
    setenv("HIPDNN_LOG_FILE", custom_log_file.c_str(), 1);
    setenv("HIPDNN_LOG_LEVEL", "info", 1);
    
    HIPDNN_LOG_INFO("Logging to custom file");
    
    _log_file = custom_log_file;
    
    verify_log_contains("Logging to custom file");
}

TEST_F(Backend_logging_test, LoggingCanBeReinitialized)
{
    setenv("HIPDNN_LOG_LEVEL", "off", 1);
    HIPDNN_LOG_INFO("This should not appear");
    
    verify_log_not_contains("This should not appear");
    
    hipdnn::logging::cleanup_logging();
    EXPECT_FALSE(hipdnn::logging::g_logging_initialized);
    
    setenv("HIPDNN_LOG_LEVEL", "info", 1);
    HIPDNN_LOG_INFO("This should appear after reinitialization");
    
    verify_log_contains("This should appear after reinitialization");
}

TEST_F(Backend_logging_test, LogPatternFormatIsCorrect)
{
    setenv("HIPDNN_LOG_LEVEL", "info", 1);
    
    HIPDNN_LOG_INFO("Pattern format test message");
    
    std::string log_content = get_log_content();
    
    // [timestamp format] [thread id] [log level] [hipdnn_backend] message
    std::regex pattern_regex(
        R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[tid \d+\] \[info\] \[hipdnn_backend\] Pattern format test message)");
    
    bool pattern_matched = false;
    std::istringstream log_stream(log_content);
    std::string line;
    
    while (std::getline(log_stream, line)) {
        if (std::regex_match(line, pattern_regex)) {
            pattern_matched = true;
            break;
        }
    }
    
    EXPECT_TRUE(pattern_matched) 
        << "Expected log format pattern not found. Log content:\n" << log_content;
}

TEST_F(Backend_logging_test, MultipleMessagesAreLogged)
{
    setenv("HIPDNN_LOG_LEVEL", "info", 1);
    
    HIPDNN_LOG_INFO("First backend message");
    HIPDNN_LOG_INFO("Second backend message");
    HIPDNN_LOG_INFO("Third backend message");
    
    std::string log_content = get_log_content();
    verify_log_contains("First backend message");
    verify_log_contains("Second backend message");
    verify_log_contains("Third backend message");
    
    // Verify expected order
    size_t pos1 = log_content.find("First backend message");
    size_t pos2 = log_content.find("Second backend message");
    size_t pos3 = log_content.find("Third backend message");
    
    EXPECT_TRUE(pos1 < pos2 && pos2 < pos3)
        << "Messages not logged in expected order";
}