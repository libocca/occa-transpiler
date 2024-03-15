#include "attributes/utils/serial_subset/common.h"
#include "pipeline/stages/transpiler/error_codes.h"

namespace oklt::serial_subset {
using namespace clang;

namespace {
const std::string externC = "extern \"C\"";
}  // namespace

HandleResult handleKernelAttribute(const Attr& a, const FunctionDecl& decl, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif

    auto& rewriter = s.getRewriter();

    // Add 'extern "C"'
    SourceRange attrRange = getAttrFullSourceRange(a);
    rewriter.ReplaceText(attrRange, externC);

    // Convert a non-pointer params to references
    auto& ctx = s.getCompiler().getASTContext();
    for (const auto param : decl.parameters()) {
        if (!param || !param->getType().getTypePtrOrNull()) {
            continue;
        }

        auto t = param->getType();
        if (!t->isPointerType()) {
            auto locRange = param->DeclaratorDecl::getSourceRange();
            rewriter.InsertTextAfter(locRange.getEnd(), " &");
        }
    }

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    if (!sema.getParsingKernelInfo() && sema.getParsingKernelInfo()->kernInfo) {
        return tl::unexpected<Error>(makeError(
            OkltTranspilerErrorCode::INTERNAL_ERROR_KERNEL_INFO_NULL, "handleKernelAttribute"));
    }
    sema.getParsingKernelInfo()->kernInfo->name = decl.getNameAsString();

    return {};
}

}  // namespace oklt::serial_subset
