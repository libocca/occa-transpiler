#include "oklt/core/transpile.h"
#include "oklt/core/ast_traversal/transpile_frontend_action.h"
//TODO: needs implementation
//#include <oklt/normalizer/Normalize.h>
//#include <oklt/normalizer/GnuAttrBasedNormalizer.h>
//#include <oklt/normalizer/MarkerBasedNormalizer.h>

#include <llvm/Support/raw_os_ostream.h>
#include <clang/Tooling/Tooling.h>

#include <fstream>

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

namespace okl {
bool transpile(std::ostream &error_stream,
               const std::filesystem::path &source_file,
               const std::filesystem::path &output_file,
               TRANSPILER_TYPE targetBackend,
               bool need_normalization)
{
    if(!std::filesystem::exists(source_file)) {
        error_stream << "File not found" << std::endl;
        return false;
    }
    std::shared_ptr<PCHContainerOperations> PCHContainerOps = std::make_shared<PCHContainerOperations>();

    Twine tool_name = "okl-transpiler";
    std::string rawFileName = source_file.filename().string();
    Twine file_name(rawFileName);
    std::vector<std::string> args = {
        "-std=c++17",
        "-fparse-all-comments",
        "-I."
    };

    std::string sourceCode;
    if(need_normalization) {
//TODO: needs implementation
//        //TODO add option for nomalizer method
//        //TODO error handing
//        sourceCode = okl::apply_gnu_attr_based_normalization(source_file).get();
//        //sourceCode = okl::normalize(source_file);
    } else {
        std::ifstream ifs(source_file.string());
        if(ifs) {
            sourceCode = {std::istreambuf_iterator<char>(ifs), {}};
        } else {
            return false;
        }
    }

    Twine code(sourceCode);
    std::ofstream ofs(output_file.string());
    std::unique_ptr<oklt::TranspileFrontendAction> action = std::make_unique<oklt::TranspileFrontendAction>(targetBackend, ofs);

    return runToolOnCodeWithArgs(std::move(action),
                                 code,
                                 args,
                                 file_name,
                                 tool_name,
                                 std::move(PCHContainerOps));
}
}

