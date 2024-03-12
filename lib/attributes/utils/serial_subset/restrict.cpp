#include "attributes/utils/serial_subset/common.h"

namespace oklt::serial_subset {
using namespace clang;

namespace {
const std::string restrictText = "__restrict__ ";
}  // namespace

HandleResult handleRestrictAttribute(const clang::Attr& a,
                                     const clang::ParmVarDecl& decl,
                                     SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif

    removeAttribute(a, s);
    s.getRewriter().InsertTextBefore(decl.getLocation(), restrictText);
    return {};
}

}  // namespace oklt::serial_subset
