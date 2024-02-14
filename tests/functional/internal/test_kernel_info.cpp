#include <gtest/gtest.h>
#include <oklt/core/metadata/program.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include "common/data_directory.h"

using namespace oklt;
using namespace oklt::tests;
using json = nlohmann::json;

TEST(TestKernelInfo, DatatypeInfoJsonTest) {
  auto dataDir = DataRootHolder::instance().dataRoot;
  auto miscDir = dataDir / "misc";
  auto datatypeJsonPath = miscDir / "datatype_info.json";
  std::ifstream datatypeJsonFile(datatypeJsonPath);
  json tests = json::parse(datatypeJsonFile);

  std::vector<DataType> deserialized;
  for (const auto& obj : tests) {
    DataType dt;
    obj.get_to(dt);
    deserialized.emplace_back(std::move(dt));
  }
  EXPECT_EQ(tests.size(), deserialized.size());
  EXPECT_EQ(4, deserialized.size()) << "Test contains different amount of entries";

  EXPECT_EQ(DatatypeCategory::BUILTIN, deserialized[0].type);
  EXPECT_STREQ("int", deserialized[0].name.c_str());

  EXPECT_EQ(DatatypeCategory::BUILTIN, deserialized[1].type);
  EXPECT_STREQ("float", deserialized[1].name.c_str());

  EXPECT_EQ(DatatypeCategory::CUSTOM, deserialized[2].type);
  EXPECT_STREQ("none", deserialized[2].name.c_str());
  EXPECT_EQ(0, deserialized[2].bytes);

  EXPECT_EQ(DatatypeCategory::CUSTOM, deserialized[3].type);
  EXPECT_STREQ("none", deserialized[3].name.c_str());
  EXPECT_EQ(4, deserialized[3].bytes);

  std::string originFormatedJson = tests.dump();
  json testedJson(deserialized);
  std::string testedFormatedJson = testedJson.dump();
  EXPECT_EQ(originFormatedJson, testedFormatedJson);
}

TEST(TestKernelInfo, ArgumentInfoJsonTest) {
  auto dataDir = DataRootHolder::instance().dataRoot;
  auto miscDir = dataDir / "misc";
  auto argsJsonPath = miscDir / "argument_info.json";
  std::ifstream argsJsonFile(argsJsonPath);
  json tests = json::parse(argsJsonFile);

  std::vector<ArgumentInfo> deserialized;
  for (const auto& obj : tests) {
    ArgumentInfo dt;
    obj.get_to(dt);
    deserialized.emplace_back(std::move(dt));
  }

  EXPECT_EQ(tests.size(), deserialized.size());
  EXPECT_EQ(4, deserialized.size()) << "Test contains different amount of entries";

  EXPECT_TRUE(deserialized[0].is_const);
  EXPECT_STREQ("entries", deserialized[0].name.c_str());
  EXPECT_FALSE(deserialized[0].is_ptr);

  EXPECT_FALSE(deserialized[1].is_const);
  EXPECT_STREQ("a", deserialized[1].name.c_str());
  EXPECT_TRUE(deserialized[1].is_ptr);

  EXPECT_FALSE(deserialized[2].is_const);
  EXPECT_STREQ("b", deserialized[2].name.c_str());
  EXPECT_TRUE(deserialized[2].is_ptr);

  EXPECT_FALSE(deserialized[3].is_const);
  EXPECT_STREQ("ab", deserialized[3].name.c_str());
  EXPECT_TRUE(deserialized[3].is_ptr);

  std::string originFormatedJson = tests.dump();
  json testedJson(deserialized);
  std::string testedFormatedJson = testedJson.dump();
  EXPECT_EQ(originFormatedJson, testedFormatedJson);
}

TEST(TestKernelInfo, KernelMetaJsonTest) {
  auto dataDir = DataRootHolder::instance().dataRoot;
  auto miscDir = dataDir / "misc";
  auto kernelMetaJsonPath = miscDir / "kernel_meta.json";
  std::ifstream kernelMetaJsonFile(kernelMetaJsonPath);
  json test = json::parse(kernelMetaJsonFile);

  KernelInfo kernelMeta;
  EXPECT_NO_THROW(test.get_to(kernelMeta));

  EXPECT_EQ(4, kernelMeta.args.size());
  EXPECT_EQ(std::string("_occa_addVectors_0"), kernelMeta.name);
}

TEST(TestKernelInfo, KernelInfoJsonTest) {
  auto dataDir = DataRootHolder::instance().dataRoot;
  auto miscDir = dataDir / "misc";
  auto kernelInfoJsonPath = miscDir / "kernel_info.json";
  std::ifstream kernelInfoJsonFile(kernelInfoJsonPath);
  json test = json::parse(kernelInfoJsonFile);

  ProgramMetaData info;
  EXPECT_NO_THROW(test.get_to(info));
  EXPECT_FALSE(info.dependencies.has_value());
  EXPECT_EQ(std::string("28e58002e281fe6b18589982a2871141d4b0bea8d0691c6564edf0d25c583d93"),
            info.hash);
  EXPECT_EQ(1, info.kernels.size());
  EXPECT_EQ(std::string("_occa_addVectors_0"), info.kernels[0].name);
  EXPECT_TRUE(info.props.has_value());
  EXPECT_EQ(std::string("nvcc"), info.props.value().compiler);
}
