#include "attributes/backend/openmp/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string prefixExprText = "\n#pragma omp atomic\n";
const std::string prefixCompoundText = "\n#pragma omp critical\n";

HandleResult handleOPENMPAtomicAttribute(SessionStage& s, const Stmt& stmt, const Attr& a) {
    SPDLOG_DEBUG("Handle [@atomic] attribute");
    auto& rewriter = s.getRewriter();

    SourceRange attrRange = getAttrFullSourceRange(a);
    if (isa<Expr>(stmt)) {
        rewriter.ReplaceText(attrRange, prefixExprText);
        return {};
    }
    if (isa<CompoundStmt>(stmt)) {
        rewriter.ReplaceText(attrRange, prefixCompoundText);
        return {};
    }

    rewriter.RemoveText(attrRange);
    return {};
}

__attribute__((constructor)) void registerOPENMPAtomicHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        TargetBackend::OPENMP, ATOMIC_ATTR_NAME, handleOPENMPAtomicAttribute);

    if (!ok) {
        SPDLOG_ERROR("[OPENMP] Failed to register {} attribute handler", ATOMIC_ATTR_NAME);
    }
}
}  // namespace
