#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string outerLoopText = "\nint _occa_exclusive_index;";
const std::string exlusiveExprText = "[_occa_exclusive_index]";

HandleResult handleOPENMPExclusiveDeclAttribute(const Attr& a,
                                                const VarDecl& decl,
                                                SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr.getNormalizedFullName() << '\n';
#endif
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
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

    auto trans =
        TranspilationBuilder(s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 1);

    SourceRange attrRange = getAttrFullSourceRange(a);
    trans.addReplacement(OKL_TRANSPILED_ATTR, attrRange, "");

    if (loopInfo->vars.exclusive.empty()) {
        auto indexLoc = compStmt->getLBracLoc().getLocWithOffset(1);
        trans.addReplacement(OKL_TRANSPILED_ATTR, indexLoc, outerLoopText);
    }

    // Process later when processing ForStmt
    loopInfo->vars.exclusive.emplace_back(std::ref(decl));

    return trans.build();
}

HandleResult handleOPENMPExclusiveExprAttribute(const Attr& a,
                                                const DeclRefExpr& expr,
                                                SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr.getNormalizedFullName() << '\n';
#endif
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (!loopInfo) {
        return tl::make_unexpected(
            Error{{}, "@exclusive: failed to fetch loop meta data from sema"});
    }

    auto loc = expr.getLocation().getLocWithOffset(expr.getNameInfo().getAsString().size());
    return TranspilationBuilder(s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 1)
        .addReplacement(OKL_EXCLUSIVE_OP, loc, exlusiveExprText)
        .build();
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
