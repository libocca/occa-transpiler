#include <oklt/core/kernel_metadata.h>
#include "util/string_utils.hpp"

#include "attributes/backend/metal/common.h"
#include "core/rewriter/rewriter_proxy.h"
#include "core/transpiler_session/attributed_type_map.h"
#include "core/utils/type_converter.h"
#include "pipeline/core/error_codes.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string FUNC_PREFIX = "kernel";

std::string getFunctionName(const FunctionDecl& func, size_t n) {
    return util::fmt("_occa_{}_{}", func.getNameAsString(), n).value();
}

std::string getFunctionParamStr(SessionStage& stage,
                                const FunctionDecl& func,
                                KernelInfo& kernelInfo) {
    std::stringstream out;

    auto& r = stage.getRewriter();
    auto& m = stage.tryEmplaceUserCtx<AttributedTypeMap>();

    kernelInfo.args.reserve(func.getNumParams() + 2);

    out << "(";

    size_t n = 0;
    for (auto p : func.parameters()) {
        if (!p) {
            continue;
        }

        auto qt = p->getType();
        std::string typeStr = qt.getNonReferenceType().getAsString();
        if (qt->isPointerType()) {
            auto qtStr = getCleanTypeString(
                QualType(qt.getNonReferenceType().getTypePtr()->getUnqualifiedDesugaredType(), 0));
            typeStr = util::fmt("device {}", qtStr).value();
        } else {
            qt.removeLocalConst();
            auto qtStr = getCleanTypeString(
                QualType(qt.getNonReferenceType().getTypePtr()->getUnqualifiedDesugaredType(), 0));
            typeStr =
                util::fmt("constant {} {}", qtStr, (qt.getTypePtrOrNull() ? "&" : "*")).value();
        }

        if (m.has(func.getASTContext(), qt, {RESTRICT_ATTR_NAME})) {
            typeStr += " __restrict__";
        }

        if (n != 0) {
            out << ", ";
        }

        out << util::fmt("{} {} [[buffer({})]]", typeStr, p->getNameAsString(), n).value();

        ++n;
    }

    if (n != 0) {
        out << ", ";
    }

    out << util::fmt(
               "{} {} [[{}]]", "uint3", "_occa_group_position", "threadgroup_position_in_grid")
               .value();
    out << ", ";
    out << util::fmt(
               "{} {} [[{}]]", "uint3", "_occa_thread_position", "thread_position_in_threadgroup")
               .value();

    out << ")";

    return out.str();
}

HandleResult handleKernelAttribute(SessionStage& s,
                                   const clang::FunctionDecl& func,
                                   const clang::Attr& a) {
    SPDLOG_DEBUG("Handle [@kernel] attribute for function '{}'", func.getNameAsString());

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto& rewriter = s.getRewriter();

    auto oklKernelInfo = toOklKernelInfo(func);
    if (!sema.getParsingKernelInfo() || !oklKernelInfo) {
        return tl::make_unexpected(
            Error{OkltPipelineErrorCode::INTERNAL_ERROR_KERNEL_INFO_NULL, "handleKernelAttribute"});
    }

    auto kernelInfo = *sema.getParsingKernelInfo();
    auto& kernels = sema.getProgramMetaData().kernels;

    auto typeStr = rewriter.getRewrittenText(func.getReturnTypeSourceRange());
    auto paramStr = getFunctionParamStr(s, func, oklKernelInfo.value());

    if (auto verified = verifyLoops(s, kernelInfo); !verified) {
        return tl::make_unexpected(std::move(verified.error()));
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

        handleChildAttr(s, child->stmt, MAX_INNER_DIMS_NAME);

        std::stringstream out;
        if (n != 0) {
            out << "}\n\n";
        }
        out << FUNC_PREFIX << " ";
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
    auto ok = registerBackendHandler(TargetBackend::METAL, KERNEL_ATTR_NAME, handleKernelAttribute);

    if (!ok) {
        SPDLOG_ERROR("[METAL] Failed to register {} attribute handler", KERNEL_ATTR_NAME);
    }
}
}  // namespace
