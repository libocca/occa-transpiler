#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/kernel_utils.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/type_converter.h"
#include "oklt/util/string_utils.h"
#include "pipeline/stages/transpiler/error_codes.h"

#include <spdlog/spdlog.h>

namespace {
using namespace clang;
using namespace oklt;

const std::string KERNEL_DEFINITION = "extern \"C\" __global__";
const std::string KERNEL_BOUNDS = "__launch_bounds__({})";

std::string getFunctionName(const FunctionDecl& func, size_t n) {
    return util::fmt("_occa_{}_{}", func.getNameAsString(), n).value();
}

std::string getFunctionAttributesStr([[maybe_unused]] const FunctionDecl& func, OklLoopInfo* info) {
    std::stringstream out;
    out << KERNEL_DEFINITION;

    if (info) {
        auto sizes = info->getInnerSizes();
        if (!sizes.hasNullOpts()) {
            auto prod = sizes.product();
            out << " " << util::fmt(KERNEL_BOUNDS, prod).value();
        }
    }

    out << " ";
    return out.str();
}

std::string getFunctionParamStr(const FunctionDecl& func, Rewriter& r) {
    auto typeLoc = func.getFunctionTypeLoc();
    return r.getRewrittenText(typeLoc.getParensRange());
}

}  // namespace

namespace oklt::cuda_subset {
using namespace clang;

HandleResult handleKernelAttribute(const Attr& a, const FunctionDecl& func, SessionStage& s) {
    SPDLOG_DEBUG("Handle [@kernel] attribute for function '{}'", func.getNameAsString());

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto& rewriter = s.getRewriter();

    auto oklKernelInfo = toOklKernelInfo(func);
    if (!sema.isParsingOklKernel() || !oklKernelInfo) {
        return tl::make_unexpected(Error{OkltTranspilerErrorCode::INTERNAL_ERROR_KERNEL_INFO_NULL,
                                         "handleKernelAttribute"});
    }

    auto kernelInfo = *sema.getParsingKernelInfo();
    auto& kernels = sema.getProgramMetaData().kernels;

    auto typeStr = rewriter.getRewrittenText(func.getReturnTypeSourceRange());
    auto paramStr = getFunctionParamStr(func, rewriter);

    if (auto verified = verifyLoops(kernelInfo); !verified) {
        return verified;
    }

    auto startPos = getAttrFullSourceRange(a).getBegin();
    size_t n = 0;
    for (auto* child : kernelInfo.topLevelOuterLoops) {
        if (!child) {
            continue;
        }
        kernels.push_back(oklKernelInfo.value());
        auto& meta = kernels.back();
        meta.name = getFunctionName(func, n);

        std::stringstream out;
        if (n != 0) {
            out << "}\n\n";
        }
        out << getFunctionAttributesStr(func, child);
        out << typeStr << " " << getFunctionName(func, n) << paramStr << " {\n";

        auto endPos = getAttrFullSourceRange(*child->attr).getBegin();
        rewriter.ReplaceText(SourceRange{startPos, endPos}, out.str());

        auto body = dyn_cast_or_null<CompoundStmt>(child->stmt.getBody());
        startPos = (body ? body->getEndLoc() : child->stmt.getRParenLoc()).getLocWithOffset(1);
        ++n;
    }

    rewriter.ReplaceText(SourceRange{startPos, func.getEndLoc()}, "\n}\n");

    return {};
}

}  // namespace oklt::cuda_subset
