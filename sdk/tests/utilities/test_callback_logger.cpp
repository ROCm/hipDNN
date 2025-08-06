// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <mutex>
#include <regex>
#include <spdlog/spdlog.h>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_sdk/logging/callback_types.h>
#include <hipdnn_sdk/logging/logger.hpp>

static std::vector<std::string> g_captured_logs;
static std::mutex g_log_mutex;

// Custom callback for testing. It doesn't fully simulate the real logging behavior,
// but the backend tests use the true callback function. The test could use the backend callback, but then it has to link against the backend.
void test_logging_callback(hipdnnSeverity_t, const char* msg)
{
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if(msg)
    {
        g_captured_logs.emplace_back(msg);
    }
}

class Callback_logger_test : public ::testing::Test
{
protected:
    const std::string _test_logger_name = COMPONENT_NAME;

    void SetUp() override
    {
        g_captured_logs.clear();

        spdlog::drop_all();

        hipdnn::logging::initialize_callback_logging(_test_logger_name, test_logging_callback);

        auto test_logger = spdlog::get(_test_logger_name);
        ASSERT_NE(test_logger, nullptr);
        test_logger->set_level(spdlog::level::trace);
    }

    void TearDown() override
    {
        spdlog::shutdown();
    }

    std::vector<std::string> get_captured_logs()
    {
        spdlog::shutdown(); // block until async queue is fully processed
        return g_captured_logs;
    }
};

TEST_F(Callback_logger_test, InfoMessageIsCorrectlyPassedToCallback)
{
    std::string test_message = "Test info message";
    HIPDNN_LOG_INFO(test_message);

    auto logs = get_captured_logs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_NE(logs[0].find(test_message), std::string::npos);
}

TEST_F(Callback_logger_test, WarnMessageIsCorrectlyPassedToCallback)
{
    std::string test_message = "Test warning message";
    HIPDNN_LOG_WARN(test_message);

    auto logs = get_captured_logs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_NE(logs[0].find(test_message), std::string::npos);
}

TEST_F(Callback_logger_test, ErrorMessageIsCorrectlyPassedToCallback)
{
    std::string test_message = "Test error message";
    HIPDNN_LOG_ERROR(test_message);

    auto logs = get_captured_logs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_NE(logs[0].find(test_message), std::string::npos);
}

TEST_F(Callback_logger_test, FormattedMessagesAreCorrectlyPassed)
{
    int value = 42;
    std::string text = "formatted";
    HIPDNN_LOG_INFO("Test {} message with value {}", text, value);

    auto logs = get_captured_logs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_NE(logs[0].find("Test formatted message with value 42"), std::string::npos);
}

TEST_F(Callback_logger_test, LogLevelsAreRespected)
{
    auto test_logger = spdlog::get(_test_logger_name);
    ASSERT_NE(test_logger, nullptr);

    // Set level to error so info and warn should be ignored
    test_logger->set_level(spdlog::level::err);

    HIPDNN_LOG_INFO("This info should not appear");
    HIPDNN_LOG_WARN("This warning should not appear");
    HIPDNN_LOG_ERROR("This error should appear");

    auto logs = get_captured_logs();
    ASSERT_EQ(logs.size(), 1);
    EXPECT_NE(logs[0].find("This error should appear"), std::string::npos);
}

TEST_F(Callback_logger_test, MultipleMessagesAreLogged)
{
    HIPDNN_LOG_INFO("First message");
    HIPDNN_LOG_WARN("Second message");
    HIPDNN_LOG_ERROR("Third message");

    auto logs = get_captured_logs();
    ASSERT_EQ(logs.size(), 3);
    EXPECT_NE(logs[0].find("First message"), std::string::npos);
    EXPECT_NE(logs[1].find("Second message"), std::string::npos);
    EXPECT_NE(logs[2].find("Third message"), std::string::npos);
}

TEST_F(Callback_logger_test, CallbackReceivesFormattedPattern)
{
    std::string test_message = "Pattern check";
    HIPDNN_LOG_INFO(test_message);

    auto logs = get_captured_logs();
    ASSERT_EQ(logs.size(), 1);

    std::regex pattern(R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[tid \d+\] \[info\] \[)"
                       + _test_logger_name + R"(\] )" + test_message);

    EXPECT_TRUE(std::regex_search(logs[0], pattern))
        << "Log message did not match expected pattern.\n"
        << "Actual log: " << logs[0];
}
