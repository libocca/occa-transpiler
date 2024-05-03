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

const std::string EXTERN_C = "extern \"C\"";
const std::string INNER_SIZES_FMT = "[[sycl::reqd_work_group_size({},{},{})]]";
const std::string SIMD_LEGHT_FMT = "[[intel::reqd_sub_group_size({})]]";
const std::string SUBMIT_QUEUE =
    R"(queue_->submit(
    [&](sycl::handler & handler_) {
      handler_.parallel_for(
        *range_,
        [=](sycl::nd_item<3> item_))";
const std::string suffixCode =
    R"(
        }
      );
    }
  );
)";

std::string getFunctionName(const FunctionDecl& func, size_t n) {
    return util::fmt("_occa_{}_{}", func.getNameAsString(), n).value();
}

std::string genFunctionSimdLengthStr([[maybe_unused]] const FunctionDecl& func, OklLoopInfo* info) {
    if (!info || info->simdLength.value_or(-1) <= 0) {
        return "";
    }

    std::stringstream out;
    out << " " << util::fmt(SIMD_LEGHT_FMT, info->simdLength.value()).value();
    return out.str();
}

std::string getFunctionAttributesStr([[maybe_unused]] const FunctionDecl& func, OklLoopInfo* info) {
    std::stringstream out;
    out << EXTERN_C;

    if (info) {
        auto sizes = info->getInnerSizes();
        if (!sizes.hasNullOpts()) {
            // NOTE: 2,1,0, since in DPCPP for some reason mapping is 0 -> Axis::Z. Also, refer to
            // axisToStr in dpcpp/common.cpp
            out << " " << util::fmt(INNER_SIZES_FMT, *sizes[2], *sizes[1], *sizes[0]).value();
        }
    }

    out << " ";
    return out.str();
}

std::string getFunctionParamStr(const FunctionDecl& func, KernelInfo& kernelInfo, oklt::Rewriter& r) {
    std::stringstream out;

    kernelInfo.args.clear();
    kernelInfo.args.reserve(func.getNumParams() + 2);

    kernelInfo.args.emplace_back(
        ArgumentInfo{.is_const = false,
                     .dtype = DataType{.typeCategory = DatatypeCategory::CUSTOM},
                     .name = "queue_",
                     .is_ptr = true});
    out << util::fmt("{} {} {}", "sycl::queue", "*", "queue_").value();

    out << ", ";

    kernelInfo.args.emplace_back(
        ArgumentInfo{.is_const = false,
                     .dtype = DataType{.typeCategory = DatatypeCategory::CUSTOM},
                     .name = "range_",
                     .is_ptr = true});
    out << util::fmt("{} {} {}", "sycl::nd_range<3>", "*", "range_").value();
    if (func.getNumParams() > 0) {
        out << ", ";
    }

    auto typeLoc = func.getFunctionTypeLoc();
    r.InsertTextAfterToken(typeLoc.getLParenLoc(), out.str());

    for (auto param : func.parameters()) {
        if (!param) {
            continue;
        }
        if (auto arg = toOklArgInfo(*param)) {
            kernelInfo.args.emplace_back(std::move(arg.value()));
        }
    }

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
            out << suffixCode;
            out << "}\n\n";
        }
        out << getFunctionAttributesStr(func, child);
        out << typeStr << " " << getFunctionName(func, n) << paramStr << " {\n";
        out << SUBMIT_QUEUE << genFunctionSimdLengthStr(func, child) << " {\n";

        auto endPos = getAttrFullSourceRange(*child->attr).getBegin().getLocWithOffset(-1);
        rewriter.ReplaceText(SourceRange{startPos, endPos}, out.str());

        auto body = dyn_cast_or_null<CompoundStmt>(child->stmt.getBody());
        startPos = (body ? body->getEndLoc() : child->stmt.getRParenLoc()).getLocWithOffset(1);

        ++n;
    }

    rewriter.ReplaceText(SourceRange{startPos, func.getEndLoc()}, suffixCode + "\n}\n");

    return {};
}

__attribute__((constructor)) void registerKernelHandler() {
    auto ok = registerBackendHandler(TargetBackend::DPCPP, KERNEL_ATTR_NAME, handleKernelAttribute);

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register {} attribute handler", KERNEL_ATTR_NAME);
    }
}
}  // namespace
