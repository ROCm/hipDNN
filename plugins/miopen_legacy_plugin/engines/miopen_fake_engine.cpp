// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "miopen_fake_engine.hpp"

Miopen_fake_engine::Miopen_fake_engine(int64_t id)
    : _id(id)
{
}

int64_t Miopen_fake_engine::id() const
{
    return _id;
}

bool Miopen_fake_engine::is_applicable(const hipdnnPluginConstData_t*) const
{
    return true;
}

size_t Miopen_fake_engine::get_workspace_size() const
{
    return 1337;
}
