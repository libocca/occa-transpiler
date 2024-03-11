#include "attributes/backend/openmp/common.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string outerLoopText = "\nint _occa_exclusive_index;";
const std::string exlusiveExprText = "[_occa_exclusive_index]";

HandleResult handleOPENMPExclusiveDeclAttribute(const Attr& a,
                                                const VarDecl& decl,
                                                SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (!loopInfo) {
        return tl::make_unexpected(
            Error{{}, "@exclusive: failed to fetch loop meta data from sema"});
    }

    auto compStmt = dyn_cast_or_null<CompoundStmt>(loopInfo->stmt.getBody());
    if (!compStmt || !loopInfo->isOuter()) {
        return tl::make_unexpected(
            Error{{}, "Must define [@exclusive] variables between [@outer] and [@inner] loops"});
    }

    auto child = loopInfo->getFirstAttributedChild();
    if (!child || !child->isInner()) {
        return tl::make_unexpected(
            Error{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"});
    }

    auto& loopInfoEx = openmp::getBackendCtxFromStage(s).getLoopInfo(loopInfo);
    auto& rewriter = s.getRewriter();

    SourceRange attrRange = getAttrFullSourceRange(a);
    rewriter.RemoveText(attrRange);

    if (loopInfoEx.exclusive.empty()) {
        auto indexLoc = compStmt->getLBracLoc().getLocWithOffset(1);
        rewriter.InsertTextAfter(indexLoc, outerLoopText);
    }

    // Process later when processing ForStmt
    loopInfoEx.exclusive.emplace_back(std::ref(decl));

    // Find max size of inner loops
    size_t sz = 0;
    for (auto child : loopInfo->children) {
        auto v = child.getSize();
        if (!v.has_value()) {
            sz = 1024;
            break;
        }
        sz = std::max(v.value(), sz);
    }
    std::string varSuffix = "[" + std::to_string(sz) + "]";

    // Add size and wrap initialization.
    auto nameLoc = decl.getLocation().getLocWithOffset(decl.getName().size());
    rewriter.InsertTextAfter(nameLoc, varSuffix);
    if (decl.hasInit()) {
        auto expr = decl.getInit();
        rewriter.InsertTextBefore(expr->getBeginLoc(), "{");
        rewriter.InsertTextAfter(decl.getEndLoc().getLocWithOffset(1), "}");
    }

    return {};
}

HandleResult handleOPENMPExclusiveExprAttribute(const Attr& a,
                                                const DeclRefExpr& expr,
                                                SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (!loopInfo) {
        return tl::make_unexpected(
            Error{{}, "@exclusive: failed to fetch loop meta data from sema"});
    }

    auto loc = expr.getLocation().getLocWithOffset(expr.getNameInfo().getAsString().size());
    s.getRewriter().InsertTextAfter(loc, exlusiveExprText);

    return {};
}

__attribute__((constructor)) void registerOPENMPExclusiveHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(handleOPENMPExclusiveExprAttribute));
    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(handleOPENMPExclusiveDeclAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << EXCLUSIVE_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace
