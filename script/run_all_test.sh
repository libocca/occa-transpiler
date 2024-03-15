#!/usr/bin/env bash

./build_Release/bin/occa-transpiler-tests --suite ./tests/functional/configs/test_suite_transpiler/backends/hip  --data_root ./tests/functional/data
./build_Release/bin/occa-transpiler-tests --suite ./tests/functional/configs/test_suite_transpiler/backends/cuda  --data_root ./tests/functional/data
./build_Release/bin/occa-transpiler-tests --suite ./tests/functional/configs/test_suite_transpiler/backends/dpcpp  --data_root ./tests/functional/data
./build_Release/bin/occa-transpiler-tests --suite ./tests/functional/configs/test_suite_transpiler/backends/openmp  --data_root ./tests/functional/data
./build_Release/bin/occa-transpiler-tests --suite ./tests/functional/configs/test_suite_transpiler/backends/serial  --data_root ./tests/functional/data

./build_Release/bin/occa-transpiler-tests --suite ./tests/functional/configs/test_suite_transpiler/common/dim  --data_root ./tests/functional/data
./build_Release/bin/occa-transpiler-tests --suite ./tests/functional/configs/test_suite_transpiler/common/kernel_metadata  --data_root ./tests/functional/data