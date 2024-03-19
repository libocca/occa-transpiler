#include <oklt/core/error.h>

#include "core/ast_traversal/transpile_frontend_action.h"
#include "core/transpiler_session/session_result.h"
#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpiler_session.h"

#include <clang/Tooling/Tooling.h>

#include <llvm/Support/raw_os_ostream.h>
#include <oklt/util/format.h>

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

namespace oklt {

TranspilerSessionResult runTranspilerStage(SharedTranspilerSession session) {
    auto& input = session->input;

    Twine tool_name = "okl-transpiler";
    // INFO: hot fix for *.okl extention
    std::string rawFileName = "main_kernel.cpp";
    Twine file_name(rawFileName);
    std::vector<std::string> args = {"-std=c++17", "-fparse-all-comments", "-I."};

    for (const auto& define : input.defines) {
        std::string def = "-D" + define;
        args.push_back(std::move(def));
    }

    for (const auto& includePath : input.inlcudeDirectories) {
        std::string incPath = "-I" + includePath.string();
        args.push_back(std::move(incPath));
    }

    Twine code(input.sourceCode);

    bool ret = runToolOnCodeWithArgs(std::make_unique<oklt::TranspileFrontendAction>(*session),
                                     code,
                                     args,
                                     file_name,
                                     tool_name,
                                     std::make_shared<PCHContainerOperations>());
    if (!ret || !session->getErrors().empty()) {
        return tl::make_unexpected(std::move(session->getErrors()));
    }

    session->output.kernel.sourceCode = oklt::format(session->output.kernel.sourceCode);

    if (!session->output.launcher.sourceCode.empty()) {
        session->output.launcher.sourceCode = oklt::format(session->output.launcher.sourceCode);
    }

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "stage 3 cpp source:\n\n" << session->output.kernel.sourceCode << '\n';
#endif

    return session;
}
}  // namespace oklt
