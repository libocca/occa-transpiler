#include <gtest/gtest.h>
#include <argparse/argparse.hpp>
#include <iostream>
#include "common/data_directory.h"

namespace fs = std::filesystem;
using namespace oklt::tests;

int main(int argc, char* argv[]) {
  // TODO: make just argument filtering function
  argparse::ArgumentParser program("occa-transpiler-tests");
  program.add_argument("-s", "--suite").default_value("").help("set suite path");
  program.add_argument("-d", "--data_root").default_value("").help("set data root folder");
  try {
    program.parse_args(argc, argv);
    auto suite = program.get<std::string>("--suite");
    std::filesystem::path suitePath(suite);
    if (!fs::exists(suitePath)) {
      std::cerr << "Can't find suite folder path" << std::endl;
      return 1;
    }
    DataRootHolder::instance().suitePath = suitePath;
    auto dataPath = program.get<std::string>("--data_root");
    std::filesystem::path p(dataPath);
    if (!fs::exists(p) && fs::is_directory(p)) {
      std::cerr << "Provided invalid data_root" << std::endl;
      return 1;
    }
    DataRootHolder::instance().dataRoot = p;
  } catch (const std::exception& err) {
    std::cerr << "Tests config error: " << err.what() << std::endl;
    std::cerr << program.usage() << std::endl;
    // return 0;
  }

  ::testing::InitGoogleTest(&argc, argv);
  auto tests_res = RUN_ALL_TESTS();
  return tests_res;
}
