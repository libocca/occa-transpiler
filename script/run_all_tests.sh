#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
SUITES_PATH=$SCRIPT_DIR/../tests/functional/configs/test_suite_transpiler/
DATA_ROOT=$SCRIPT_DIR/../tests/functional/data
TEST_BINARY=$SCRIPT_DIR/../build/bin/occa-transpiler
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
    -s | --suites_path)
        SUITE="$2"
        shift # past argument
        shift # past value
        ;;
    -d | --data_root)
        DATA_ROOT="$2"
        shift # past argument
        shift # past value
        ;;
    -t | --test_binary_path)
        TEST_BINARY="$2"
        shift # past argument
        shift # past value
        ;;
    -v | --verbose)
        VERBOSE=true
        shift # past argument
        ;;
    -h | --help)
        echo "Usage: run_all_tests [-t|--test_binary_path <t>] [-s|--suites_path <s>] [-d|--data_root <d>] [-v|--verbose]"
        exit 0
        ;;
    -* | --*)
        echo "Unknown option $1"
        exit 1
        ;;
    *)
        POSITIONAL_ARGS+=("$1") # save positional arg
        shift                   # past argument
        ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

TEST_CASES=(
    'backends/hip'
    'backends/cuda'
    'backends/dpcpp'
    'backends/openmp'
    'backends/serial'
    'common/dim'
    'common/kernel_metadata'
)

for case in "${TEST_CASES[@]}"; do
    EXEC="$TEST_BINARY --suite $SUITES_PATH/$case --data_root $DATA_ROOT"
    if [[ $VERBOSE ]]; then
        echo "[VERBOSE] Run $EXEC"
    fi
    eval $EXEC
done
