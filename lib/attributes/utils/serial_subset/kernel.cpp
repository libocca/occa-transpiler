#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/type_converter.h"
#include "pipeline/core/error_codes.h"

#include <spdlog/spdlog.h>
namespace oklt::serial_subset {
using namespace clang;

namespace {
const std::string EXTERN_C = "extern \"C\"";
}  // namespace

HandleResult handleKernelAttribute(SessionStage& s, const FunctionDecl& func, const Attr& a) {
    SPDLOG_DEBUG("Handle [@kernel] attribute for function '{}'", func.getNameAsString());

    auto& rewriter = s.getRewriter();
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();

    auto oklKernelInfo = toOklKernelInfo(func);
    if (!sema.getParsingKernelInfo() && !oklKernelInfo) {
        return tl::make_unexpected(Error{OkltPipelineErrorCode::INTERNAL_ERROR_KERNEL_INFO_NULL,
                                         "handleKernelAttribute"});
    }

    // Add 'extern "C"'
    SourceRange attrRange = getAttrFullSourceRange(a);
    rewriter.ReplaceText(attrRange, EXTERN_C);

    // Convert a non-pointer params to references
    auto& ctx = s.getCompiler().getASTContext();
    for (const auto param : func.parameters()) {
        if (!param || !param->getType().getTypePtrOrNull()) {
            continue;
        }

        auto t = param->getType();
        if (!t->isPointerType()) {
            auto locRange = param->DeclaratorDecl::getSourceRange();
            rewriter.InsertTextAfter(locRange.getEnd(), " &");
        }
    }

    auto& kernels = sema.getProgramMetaData().kernels;
    kernels.push_back(oklKernelInfo.value());

    return {};
}

}  // namespace oklt::serial_subset
