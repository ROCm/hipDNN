// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/scoped_resource.hpp>

using namespace hipdnn::sdk;

namespace
{

struct Resource
{
    bool released = false;
};

} // namespace

TEST(ScopedResourceTest, BasicUsage)
{
    Resource r;

    {
        utilities::Scoped_resource sr(&r, [](auto r) { r->released = true; });
        ASSERT_TRUE(!sr.is_empty());
        ASSERT_EQ(sr.get()->released, r.released);
    }

    ASSERT_TRUE(r.released);
}

TEST(ScopedResourceTest, CheckMove)
{
    Resource r;

    {
        utilities::Scoped_resource sr1(&r, [](auto r) { r->released = true; });

        utilities::Scoped_resource sr2(std::move(sr1));
        ASSERT_TRUE(sr1.is_empty());
        ASSERT_FALSE(sr2.is_empty());
        ASSERT_FALSE(r.released);
    }

    ASSERT_TRUE(r.released);
}

TEST(ScopedResourceTest, CheckMoveAssign)
{
    Resource r1;
    Resource r2;

    {
        auto dtor = [](auto r) { r->released = true; };
        utilities::Scoped_resource sr1(&r1, dtor);
        utilities::Scoped_resource sr2(&r2, dtor);
        sr2 = std::move(sr1);

        ASSERT_FALSE(r1.released);
        ASSERT_TRUE(r2.released);

        ASSERT_TRUE(sr1.is_empty());
        ASSERT_FALSE(sr2.is_empty());
    }

    ASSERT_TRUE(r1.released);
}
