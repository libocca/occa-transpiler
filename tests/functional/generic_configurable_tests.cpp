#include "common/data_directory.h"
#include "common/load_test_suites.h"
#include "oklt/core/transpile.h"
#include <gtest/gtest.h>

namespace fs = std::filesystem;
using namespace oklt::tests;

class GenericTest : public testing::TestWithParam<GenericSuite>
{};

TEST_P( GenericTest, OCCATests )
{
    GenericSuite suite = GetParam();
    for(const auto &testCase: suite.testCaseFiles) {

    }
}


INSTANTIATE_TEST_SUITE_P(
    GenericSuiteTests,
    GenericTest,
    testing::ValuesIn( loadTestsSuite() ));
