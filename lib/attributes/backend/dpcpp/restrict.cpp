#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;
const std::string RESTRICT_MODIFIER = "__restrict__";

HandleResult handleRestrictAttribute(const clang::Attr& a,
                                     const clang::Decl& decl,
                                     SessionStage& s) {
    SPDLOG_DEBUG("Handle [@restrict] attribute");

    SourceLocation identifierLoc = decl.getLocation();
    std::string restrictText = " " + RESTRICT_MODIFIER + " ";

    // INFO: might be better to use rewriter.getRewrittenText() method

    auto& ctx = decl.getASTContext();
    SourceRange r1(decl.getSourceRange().getBegin(), identifierLoc);
    auto part1 = getSourceText(r1, ctx);

    auto ident = [](const clang::Decl& decl) {
        switch (decl.getKind()) {
            case Decl::Field:
                return cast<FieldDecl>(decl).getNameAsString();
            case Decl::ParmVar:
                return cast<ParmVarDecl>(decl).getQualifiedNameAsString();
            default:
                return cast<VarDecl>(decl).getQualifiedNameAsString();
        }
    }(decl);

    std::string modifiedArgument = part1 + restrictText + ident;

    s.getRewriter().ReplaceText({getAttrFullSourceRange(a).getBegin(), decl.getEndLoc()},
                                part1 + restrictText + ident);
    return {};
}

__attribute__((constructor)) void registerCUDARestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, RESTRICT_ATTR_NAME},
        makeSpecificAttrHandle(handleRestrictAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << RESTRICT_ATTR_NAME
                     << " attribute handler for DPCPP backend\n";
    }
}
}  // namespace
