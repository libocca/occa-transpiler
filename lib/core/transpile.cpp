#include "oklt/core/transpile.h"
#include "oklt/core/ast_traversal/transpile_frontend_action.h"
//TODO: needs implementation
//#include <oklt/normalizer/Normalize.h>
//#include <oklt/normalizer/GnuAttrBasedNormalizer.h>
//#include <oklt/normalizer/MarkerBasedNormalizer.h>

#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/JSON.h>
#include <clang/Tooling/Tooling.h>

#include <fstream>

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

namespace oklt {

struct Config {
  // std::string action;
  std::string backend;
  bool normalization;
  std::vector<std::string> includes;
  std::vector<std::string> defs;
};

//TODO: change error type
tl::expected<TranspilerInput, std::string> make_transpile_input(const std::filesystem::path &sourceFile,
                                                               const std::string &json)
{
  if(!std::filesystem::exists(sourceFile)) {
    return tl::unexpected("Wrong file path");
  }
  std::ifstream sourceMapFile( sourceFile );
  std::string sourceCode {std::istreambuf_iterator<char>(sourceMapFile), {}};
  auto expectedObj = llvm::json::parse(json);

  //TODO: convert llvm::Error to interface error type
  if(!expectedObj) {
    return tl::unexpected<std::string>("Can't parse JSON");
  }
  auto obj = expectedObj.get().getAsObject();
  if(!obj) {
    return tl::unexpected<std::string>("Json is not object");
  }
  auto backendOpt = obj->getString("backend");
  if(!backendOpt) {
    return tl::unexpected<std::string>("Backend field is missing");
  }
  auto expectBackend = backendFromString(backendOpt.value().str());
  //TODO: check error cast to error interface type compatibility
  if(!expectBackend) {
    return tl::unexpected<std::string>(expectBackend.error());
  }
  auto normOpt = obj->getBoolean("normalization");
  if(!normOpt) {
    return tl::unexpected<std::string>("normalization field is missing");
  }
  return TranspilerInput {
    .sourceCode = sourceCode,
    .sourcePath = sourceFile,
    .inlcudeDirectories = {},
    .defines = {},
    .targetBackend = expectBackend.value(),
    .normalization = normOpt.value()
  };
}

tl::expected<TranspilerResult,std::vector<Error>> transpile(TranspilerInput input)
{
  Twine tool_name = "okl-transpiler";
  std::string rawFileName = input.sourcePath.filename().string();
  Twine file_name(rawFileName);
  std::vector<std::string> args = {
      "-std=c++17",
      "-fparse-all-comments",
      "-I."
  };

  std::string sourceCode;
  if(input.normalization) {
    //TODO: needs implementation
    //        //TODO add option for nomalizer method
    //        //TODO error handing
    //        sourceCode = okl::apply_gnu_attr_based_normalization(source_file).get();
    //        //sourceCode = okl::normalize(source_file);
  } else {
    sourceCode = input.sourceCode;
  }

  oklt::TranspilerSession session {input.targetBackend};

  Twine code(sourceCode);
  std::shared_ptr<PCHContainerOperations> pchOps = std::make_shared<PCHContainerOperations>();
  std::unique_ptr<oklt::TranspileFrontendAction> action =
      std::make_unique<oklt::TranspileFrontendAction>(session);

  bool ret = runToolOnCodeWithArgs(std::move(action),
                               code,
                               args,
                               file_name,
                               tool_name,
                               std::move(pchOps));
  if(!ret) {
    return tl::unexpected(std::vector<Error>{});
  }
  TranspilerResult result;
  result.kernel.outCode = session.transpiledCode;
  return result;
}
}

