#!/usr/bin/env bash

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set -e

# This script can be run from inside the tests directory, in which case the
# path to the tests directory does not need to be specified.  If the script is
# run from outside the tests directory, the path to the tests directory must
# be specified as an argument.
testSrcDir=$(pwd)
if [ "$#" -eq 1 ]; then
    testSrcDir=$1
fi

# install dependencies in a virtual environment
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install ${testSrcDir}/..
.venv/bin/pip install -r ${testSrcDir}/requirements.txt
.venv/bin/pip install pytest

# run the tests
.venv/bin/python -m pytest -v ${testSrcDir}

# cleanup
deactivate
rm -rf .venv
rm -rf __pycache__
rm -rf tests/__pycache__
rm -rf ${testSrcDir}/../build
rm -rf ${testSrcDir}/../hipdnn.egg-info