#include <oklt/core/kernel_metadata.h>
#include "util/string_utils.hpp"

#include "attributes/attribute_names.h"
#include "attributes/utils/kernel_utils.h"
#include "core/handler_manager/backend_handler.h"
#include "core/rewriter/rewriter_proxy.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/sema/okl_sema_info.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/type_converter.h"

#include "pipeline/core/error_codes.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string EXTERN_ATTRIBUTE = "__kernel ";
const std::string EXTERN_ATTRIBUTE_ADD = "__attribute__ ";
const std::string INNER_SIZES_FMT = "((reqd_work_group_size({},{},{})))";

std::string getFunctionName(const FunctionDecl& func, size_t n) {
    return util::fmt("_occa_{}_{}", func.getNameAsString(), n).value();
}

std::string getFunctionAttributesStr([[maybe_unused]] const FunctionDecl& func, OklLoopInfo* info) {
    std::stringstream out;
    out << EXTERN_ATTRIBUTE;

    if (info) {
        auto sizes = info->getInnerSizes();
        if (!sizes.hasNullOpts()) {
            out << EXTERN_ATTRIBUTE_ADD;
            out << util::fmt(INNER_SIZES_FMT, *sizes[0], *sizes[1], *sizes[2]).value();
        }
    }

    out << "\n";
    return out.str();
}

std::string getFunctionParamStr(const FunctionDecl& func, KernelInfo& kernelInfo, oklt::Rewriter& r) {
    kernelInfo.args.clear();
    kernelInfo.args.reserve(func.getNumParams());

    for (auto param : func.parameters()) {
        if (!param) {
            continue;
        }
        
        kernelInfo.args.emplace_back(toOklArgInfo(*param).value());
        auto t = param->getType();
        if (t->isPointerType()) {
            r.InsertTextBefore(param->getBeginLoc(), "__global ");
        }
    }
    
    auto typeLoc = func.getFunctionTypeLoc();
    return r.getRewrittenText(typeLoc.getParensRange());
}

HandleResult handleKernelAttribute(SessionStage& s,
                                   const clang::FunctionDecl& func,
                                   const clang::Attr& a) {
    SPDLOG_DEBUG("Handle [@kernel] attribute for function '{}'", func.getNameAsString());

    auto& rewriter = s.getRewriter();
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();

    if (!sema.getParsingKernelInfo()) {
        return tl::make_unexpected(Error{OkltPipelineErrorCode::INTERNAL_ERROR_KERNEL_INFO_NULL,
                                         "handleKernelAttribute"});
    }

    auto kernelInfo = *sema.getParsingKernelInfo();
    auto& kernels = sema.getProgramMetaData().kernels;

    auto oklKernelInfo = KernelInfo{.name = func.getNameAsString()};
    auto typeStr = rewriter.getRewrittenText(func.getReturnTypeSourceRange());
    auto paramStr = getFunctionParamStr(func, oklKernelInfo, rewriter);

    if (auto verified = verifyLoops(s, kernelInfo); !verified) {
        return tl::make_unexpected(std::move(verified.error()));
    }

    size_t n = 0;
    auto startPos = getAttrFullSourceRange(a).getBegin();
    for (auto* child : kernelInfo.topLevelOuterLoops) {
        if (!child) {
            continue;
        }

        kernels.push_back(oklKernelInfo);
        auto& meta = kernels.back();
        meta.name = getFunctionName(func, n);

        handleChildAttr(s, child->stmt, MAX_INNER_DIMS_NAME);
        
        std::stringstream out;
        if (n != 0) {
            out << "}\n\n";
        }

        out << getFunctionAttributesStr(func, child);
        out << typeStr << " " << getFunctionName(func, n) << paramStr << ";\n";
        out << "\n";

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

__attribute__((constructor)) void registerKernelHandler() {
    auto ok = registerBackendHandler(TargetBackend::OPENCL, KERNEL_ATTR_NAME, handleKernelAttribute);

    if (!ok) {
        SPDLOG_ERROR("[OPENCL] Failed to register {} attribute handler", KERNEL_ATTR_NAME);
    }
}
}  // namespace
