#include <oklt/core/error.h>

#include "core/ast_traversal/transpile_frontend_action.h"
#include "core/transpiler_session/session_result.h"
#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpiler_session.h"

#include <clang/Tooling/Tooling.h>

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

namespace oklt {

TranspilerSessionResult runTranspilerStage(SharedTranspilerSession session) {
    auto& input = session->input;

    Twine tool_name = "okl-transpiler";
    std::string rawFileName = input.sourcePath.filename().string();
    Twine file_name(rawFileName);
    std::vector<std::string> args = {"-std=c++17", "-fparse-all-comments", "-I."};

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

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "stage 3 cpp source:\n\n" << session->output.kernel.sourceCode << '\n';
#endif

    return session;
}
}  // namespace oklt
