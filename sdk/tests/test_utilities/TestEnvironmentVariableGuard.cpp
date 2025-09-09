// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/EnvironmentVariableGuard.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

using namespace hipdnn_sdk::test_utilities;

TEST(TestEnvironmentVariableGuard, RestoresOriginalValue)
{
    const char* testVar = "HIPDNN_TEST_ENV_VAR";
    const char* originalValue = "original";
    const char* newValue = "modified";

    hipdnn_sdk::utilities::setEnv(testVar, originalValue);

    {
        EnvironmentVariableGuard guard(testVar);

        hipdnn_sdk::utilities::setEnv(testVar, newValue);

        EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), newValue);
    }

    EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), originalValue);

    hipdnn_sdk::utilities::unsetEnv(testVar);
}

TEST(TestEnvironmentVariableGuard, RestoresUnsetVariable)
{
    const char* testVar = "HIPDNN_TEST_ENV_VAR_UNSET";

    hipdnn_sdk::utilities::unsetEnv(testVar);

    {
        EnvironmentVariableGuard guard(testVar);

        hipdnn_sdk::utilities::setEnv(testVar, "temporary");

        EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), "temporary");
    }

    EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), "");
    EXPECT_TRUE(hipdnn_sdk::utilities::getEnv(testVar).empty());
}

TEST(TestEnvironmentVariableGuard, HandlesEmptyValue)
{
    const char* testVar = "HIPDNN_TEST_ENV_VAR_EMPTY";

    hipdnn_sdk::utilities::setEnv(testVar, "");

    {
        EnvironmentVariableGuard guard(testVar);

        hipdnn_sdk::utilities::setEnv(testVar, "non-empty");

        EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), "non-empty");
    }

    EXPECT_EQ(hipdnn_sdk::utilities::getEnv(testVar, ""), "");

    hipdnn_sdk::utilities::unsetEnv(testVar);
}
