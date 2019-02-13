#!/bin/sh

echo "\nRunning hipDNN unit tests...\n"
./test/unittest
echo "Completed unit tests! \n\n"
echo "Comparing results with benchmarked results ...\n\n"
./test/compare_results
echo "\n\nCompleted!\n"
