#!/bin/sh

echo "\nStarting hipDNN unit tests...\n"
./test/unittest
echo "Completed unit tests! \n\n"
echo "Comparing results with GeForce GTX 980 Ti (NVidia) results ...\n\n"
./test/compare_results
echo "\n\nCompleted!\n"
