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

readarray -d '' TEST_SUITES < <(find $SUITES_PATH -name "suite.json" -print0)

SKIP_LIST=(
    'includes'
)

for suite in "${TEST_SUITES[@]}"; do
    test_dir=$(realpath $(dirname $suite))
    test_name=$(basename $test_dir)

    # If this suite is in skip list, skip it
    if [[ " ${SKIP_LIST[*]} " =~ [[:space:]]${test_name}[[:space:]] ]]; then
        continue
    fi

    EXEC="$TEST_BINARY --suite $test_dir --data_root $DATA_ROOT"
    if [[ $VERBOSE ]]; then
        echo "[VERBOSE] Run $EXEC"
    fi
    eval $EXEC
    if [ $? -ne 0 ]; then # stop if test failed
        exit 1
    fi
done
