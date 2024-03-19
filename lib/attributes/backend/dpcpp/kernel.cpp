#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/type_converter.h"
#include "pipeline/stages/transpiler/error_codes.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string externC = "extern \"C\"";
const std::string additionalArguments = "sycl::queue * queue_, sycl::nd_range<3> * range_";
const std::string prefixCode =
    R"(queue_->submit(
    [&](sycl::handler & handler_) {
      handler_.parallel_for(
        *range_,
        [=](sycl::nd_item<3> item_) {
)";
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

std::string getFunctionAttributesStr([[maybe_unused]] const FunctionDecl& func,
                                     OklLoopInfo& child) {
    std::stringstream out;
    out << externC;

    // TODO: Add  "[[sycl::reqd_work_group_size(x, y, z)]]"

    out << " ";
    return out.str();
}

std::string getFunctionParamStr(const FunctionDecl& func, KernelInfo& kernelInfo, Rewriter& r) {
    std::stringstream out;

    kernelInfo.args.clear();
    kernelInfo.args.reserve(func.getNumParams() + 2);

    kernelInfo.args.emplace_back(ArgumentInfo{.is_const = false,
                                              .dtype = DataType{.type = DatatypeCategory::CUSTOM},
                                              .name = "queue_",
                                              .is_ptr = true});
    out << util::fmt("{} {} {}", "sycl::queue", "*", "queue_").value();

    out << ", ";

    kernelInfo.args.emplace_back(ArgumentInfo{.is_const = false,
                                              .dtype = DataType{.type = DatatypeCategory::CUSTOM},
                                              .name = "range_",
                                              .is_ptr = true});
    out << util::fmt("{} {} {}", "sycl::nd_range<3>", "*", "range_").value();
    if (func.getNumParams() > 0) {
        out << ", ";
    }

    for (auto param : func.parameters()) {
        if (!param) {
            continue;
        }
        if (auto arg = toOklArgInfo(*param)) {
            kernelInfo.args.emplace_back(std::move(arg.value()));
        }
    }

    out << r.getRewrittenText(func.getParametersSourceRange());
    return out.str();
}

HandleResult handleKernelAttribute(const clang::Attr& a,
                                   const clang::FunctionDecl& func,
                                   SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @kernel attribute (DPCPP backend): return type: "
                 << func.getReturnType().getAsString()
                 << ", old kernel name: " << func.getNameAsString() << '\n';
#endif

    auto& rewriter = s.getRewriter();
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();

    if (!sema.getParsingKernelInfo()) {
        return tl::make_unexpected(Error{OkltTranspilerErrorCode::INTERNAL_ERROR_KERNEL_INFO_NULL,
                                         "handleKernelAttribute"});
    }

    auto kernelInfo = *sema.getParsingKernelInfo();
    auto& kernels = sema.getProgramMetaData().kernels;

    auto oklKernelInfo = KernelInfo{.name = func.getNameAsString()};
    auto typeStr = rewriter.getRewrittenText(func.getReturnTypeSourceRange());
    auto paramStr = getFunctionParamStr(func, oklKernelInfo, rewriter);

    size_t n = 0;
    auto startPos = getAttrFullSourceRange(a).getBegin();
    for (auto& child : kernelInfo.children) {
        kernels.push_back(oklKernelInfo);
        auto& meta = kernels.back();
        meta.name = getFunctionName(func, n);

        std::stringstream out;
        if (n != 0) {
            out << suffixCode;
            out << "}\n\n";
        }
        out << getFunctionAttributesStr(func, child);
        out << typeStr << " " << getFunctionName(func, n) << "(" << paramStr << ")"
            << " {\n";
        out << prefixCode;

        auto endPos = getAttrFullSourceRange(child.attr).getBegin().getLocWithOffset(-1);
        rewriter.ReplaceText(SourceRange{startPos, endPos}, out.str());

        auto body = dyn_cast_or_null<CompoundStmt>(child.stmt.getBody());
        startPos = (body ? body->getEndLoc() : child.stmt.getRParenLoc()).getLocWithOffset(1);

        ++n;
    }

    if (!kernelInfo.children.empty()) {
        rewriter.ReplaceText(SourceRange{startPos, func.getEndLoc()}, suffixCode + "\n}\n");
    }

    return {};
}

__attribute__((constructor)) void registerKernelHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, KERNEL_ATTR_NAME}, makeSpecificAttrHandle(handleKernelAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << KERNEL_ATTR_NAME << " attribute handler (DPCPP)\n";
    }
}
}  // namespace
