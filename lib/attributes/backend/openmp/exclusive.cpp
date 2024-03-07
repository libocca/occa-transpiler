#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string outerLoopText = "\nint _occa_exclusive_index;";
const std::string exclusiveNullText = "_occa_exclusive_index = 0;\n";
const std::string exclusiveIncText = "++_occa_exclusive_index;\n";
const std::string exlusiveExprText = "[_occa_exclusive_index]";

struct ExlusiveRecrdCtx {
    std::set<OklLoopInfo*> loopSet;
};

using LoopList = std::list<std::reference_wrapper<OklLoopInfo>>;
bool getInnerMostLoops(OklLoopInfo& loopInfo, LoopList& outList) {
    for (auto& child : loopInfo.children) {
        if (getInnerMostLoops(child, outList)) {
            continue;
        }
        if (child.metadata.isInner()) {
            outList.emplace_back(std::ref(child));
        }
    }
    return !loopInfo.children.empty();
}

HandleResult handleOPENMPExclusiveDeclAttribute(const Attr& a,
                                                const VarDecl& decl,
                                                SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto& exCtx = s.tryEmplaceUserCtx<ExlusiveRecrdCtx>();

    auto loopInfo = sema.getLoopInfo();
    if (!loopInfo) {
        return tl::make_unexpected(
            Error{{}, "@exclusive: failed to fetch loop meta data from sema"});
    }

    auto compStmt = dyn_cast_or_null<CompoundStmt>(loopInfo->stmt.getBody());
    if (!compStmt || loopInfo->metadata.type != LoopMetaType::Outer) {
        return tl::make_unexpected(
            Error{{}, "Must define [@exclusive] variables between [@outer] and [@inner] loops"});
    }

    auto firstChild = loopInfo->getFirstAttributedChild();
    if (!firstChild || firstChild->metadata.type != LoopMetaType::Inner) {
        return tl::make_unexpected(
            Error{{}, "Must define [@exclusive] variables between [@outer] and [@inner] loops"});
    }

    auto& rewriter = s.getRewriter();

    SourceRange attrRange = getAttrFullSourceRange(a);
    rewriter.RemoveText(attrRange);

    if (exCtx.loopSet.find(loopInfo) == exCtx.loopSet.end()) {
        // Initialize index variable
        auto indexLoc = compStmt->getLBracLoc().getLocWithOffset(1);
        rewriter.InsertText(indexLoc, outerLoopText, true, true);

        // Null index variable before each child
        for (auto& child : loopInfo->children) {
            SourceRange attrRange = getAttrFullSourceRange(child.attr);
            rewriter.InsertTextAfterToken(attrRange.getBegin().getLocWithOffset(-1),
                                          exclusiveNullText);
        }

        // Increase index for each innermost loop
        LoopList innerLoops;
        static_cast<void>(getInnerMostLoops(*loopInfo, innerLoops));
        for (auto& child : innerLoops) {
            auto compStmt = dyn_cast_or_null<CompoundStmt>(child.get().stmt.getBody());
            SourceLocation incLoc = compStmt ? compStmt->getRBracLoc().getLocWithOffset(-1)
                                             : child.get().stmt.getEndLoc();
            rewriter.InsertText(incLoc, exclusiveIncText, false, true);
        }
    }
    exCtx.loopSet.insert(loopInfo);

    // Calculate Inner max Loop size
    size_t sz = 0;
    for (auto& child : loopInfo->children) {
        auto v = child.getSize();
        if (!v.has_value()) {
            sz = 1024;
            break;
        }
        sz = std::max(v.value(), sz);
    }
    std::string varSuffix = "[" + std::to_string(sz) + "]";

    // Add value suffix and add parenthesis to innit
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
