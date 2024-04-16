#include "attributes/attribute_names.h"
#include "attributes/frontend/params/tile.h"
#include "attributes/utils/serial_subset/handle.h"
#include "core/attribute_manager/attr_stmt_handler.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/sema/okl_sema_info.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"
#include "core/utils/type_converter.h"
#include "pipeline/core/error_codes.h"

#include <oklt/core/kernel_metadata.h>

#include <clang/Rewrite/Core/Rewriter.h>
#include <spdlog/spdlog.h>

// #define OKL_LAUNCHER_RECURSIVE

namespace {
using namespace oklt;
using namespace clang;

const std::string includeOCCA = "<occa/core/kernel.hpp>";
const std::string externC = "extern \"C\"";

struct LoopMetaData {
    LoopTypes type = {LoopType::Regular};

    struct {
        std::string type;
        std::string name;
    } var;
    struct {
        std::string start;
        std::string end;
        size_t size = 0;
    } range;
    struct {
        std::string cmp;
        BinOp op = BinOp::Eq;
    } condition;
    struct {
        std::string val;
        union {
            UnOp uo;
            BinOp bo;
        } op;
    } inc;

    [[nodiscard]] bool IsInc() const {
        bool ret = false;
        if (inc.val.empty()) {
            ret = (inc.op.uo == UnOp::PreInc || inc.op.uo == UnOp::PostInc);
        } else {
            ret = (inc.op.bo == BinOp::AddAssign);
        }

        ret = (ret && (condition.op == BinOp::Le || condition.op == BinOp::Lt));

        return ret;
    };

    [[nodiscard]] std::string getRangeSizeStr() const {
        if (IsInc()) {
            return range.end + " - " + range.start;
        } else {
            return range.start + " - " + range.end;
        };
    };

    explicit LoopMetaData(const OklLoopInfo& l, const oklt::Rewriter& r) {
        type = l.type;
        var.type = l.var.typeName;
        var.name = l.var.name;

        // TODO: Currently getLatestSourceText failes. Possibly due to bug in Rewriter
        auto& ctx = l.var.varDecl->getASTContext();
        range.start = "(" + getLatestSourceText(*l.range.start, r) + ")";
        range.end = "(" + getLatestSourceText(*l.range.end, r) + ")";
        range.size = l.range.size;

        condition.cmp = getLatestSourceText(*l.condition.cmp, r);
        condition.op = l.condition.op;

        if (l.inc.val) {
            inc.val = "(" + getLatestSourceText(*l.inc.val, r) + ")";
            inc.op.bo = l.inc.op.bo;
        } else {
            inc.op.uo = l.inc.op.uo;
        }
    }
};

std::string getTiledVariableName(const LoopMetaData& meta) {
    return "_occa_tiled_" + meta.var.name;
}

// TODO: Replace with ArgumentInfo::toString()
std::string getFunctionParamStr(const FunctionDecl& func, KernelInfo& kernelInfo) {
    std::stringstream out;
    // out << "(";

    kernelInfo.args.clear();
    kernelInfo.args.reserve(func.getNumParams() + 1);

    kernelInfo.args.emplace_back(
        ArgumentInfo{.is_const = false,
                     .dtype = DataType{.typeCategory = DatatypeCategory::CUSTOM},
                     .name = "deviceKernels",
                     .is_ptr = true});
    out << util::fmt("{} {} {}", "occa::modeKernel_t", "**", "deviceKernels").value();

    for (auto p : func.parameters()) {
        if (!p) {
            continue;
        }
        out << ", ";

        auto t = p->getType();
        if (t.getTypePtrOrNull() && !t->isPointerType()) {
            kernelInfo.args.emplace_back(toOklArgInfo(*p).value());
            kernelInfo.args.back().is_const = true;
            out << util::fmt(
                       "{} {} {}", t.getNonReferenceType().getAsString(), "&", p->getNameAsString())
                       .value();
        } else {
            kernelInfo.args.emplace_back(
                ArgumentInfo{.is_const = false,
                             .dtype = DataType{.typeCategory = DatatypeCategory::CUSTOM},
                             .name = p->getNameAsString(),
                             .is_ptr = true});
            out << util::fmt("{} {} {}", "occa::modeMemory_t", "*", p->getNameAsString()).value();
        }
    }

    return out.str();
}

std::string getLoopInfoStr(const LoopMetaData& loop, size_t n, bool isOuter) {
    std::stringstream out;

    auto start = std::string_view{loop.range.start.data(), loop.range.start.size()};
    out << loop.var.type << " " << loop.var.name << " = " << util::unParen(start) << ";\n";
    out << (isOuter ? "outer" : "inner") << "[" << n << "] = ";

    if (!loop.inc.val.empty()) {
        out << "(";
    }

    switch (loop.condition.op) {
        case BinOp::Le:
        case BinOp::Ge:
            out << "1 + ";
            break;
        default:
            break;
    }

    out << loop.getRangeSizeStr();

    if (!loop.inc.val.empty()) {
        out << " + " << loop.inc.val << " - 1) / " << loop.inc.val;
    }

    out << ";\n";

    return out.str();
}

#ifdef OKL_LAUNCHER_RECURSIVE
void collectLoops(OklLoopInfo& loopInfo, std::list<OklLoopInfo*>& out) {
    if (!loopInfo.isRegular()) {
        out.push_back(&loopInfo);
    }
    for (auto& child : loopInfo.children) {
        if (!child.children.empty()) {
            collectLoops(child, out);
            continue;
        }
        if (!child.isRegular()) {
            out.push_back(&child);
        }
    }
}
#else
void collectLoops(OklLoopInfo& loopInfo, std::list<OklLoopInfo*>& out) {
    if (!loopInfo.isRegular()) {
        out.push_back(&loopInfo);
    }
    if (!loopInfo.children.empty()) {
        auto& child = loopInfo.children.front();
        if (!child.children.empty()) {
            collectLoops(child, out);
        } else if (!child.isRegular()) {
            out.push_back(&child);
        }
    }
}
#endif

std::pair<LoopMetaData, LoopMetaData> splitTileAttr(OklLoopInfo& loopInfo,
                                                    const oklt::Rewriter& r) {
    auto sz = util::parseStrTo<size_t>(loopInfo.tileSize);

    // Prepare first loop
    auto firstMeta = LoopMetaData(loopInfo, r);
    firstMeta.var.name = getTiledVariableName(firstMeta);
    if (sz.value_or(1024) > 0) {
        if (firstMeta.inc.val.empty()) {
            firstMeta.inc.val = loopInfo.tileSize;
            switch (firstMeta.inc.op.uo) {
                case UnOp::PreInc:
                case UnOp::PostInc:
                    firstMeta.inc.op.bo = BinOp::AddAssign;
                    break;
                case UnOp::PreDec:
                case UnOp::PostDec:
                    firstMeta.inc.op.bo = BinOp::RemoveAssign;
                    break;
            }
        } else {
            firstMeta.inc.val = "(" + loopInfo.tileSize + " * " + firstMeta.inc.val + ")";
        }
    }

    // Prepare second loop
    auto secondMeta = LoopMetaData(loopInfo, r);
    secondMeta.range.start = firstMeta.var.name;
    switch (secondMeta.condition.op) {
        case BinOp::Le:
            secondMeta.condition.op = BinOp::Lt;
            break;
        case BinOp::Ge:
            secondMeta.condition.op = BinOp::Gt;
            break;
    }
    if (sz.value_or(1024) > 0) {
        secondMeta.range.end = "(" + firstMeta.var.name + " + " + loopInfo.tileSize + ")";
    } else {
        secondMeta.range.end = firstMeta.var.name;
    }

    return {firstMeta, secondMeta};
}

std::string getRootLoopBody(const FunctionDecl& decl,
                            OklLoopInfo& loopInfo,
                            size_t loopNo,
                            SessionStage& s) {
    std::stringstream out;
    auto& r = s.getRewriter();

    out << " {\n";

    // List all loops
    std::list<OklLoopInfo*> loops = {};
    collectLoops(loopInfo, loops);

    // Prepare metadata for outer and inner loops
    std::list<LoopMetaData> outer = {};
    std::list<LoopMetaData> inner = {};
    for (auto child : loops) {
        if (loopInfo.isRegular()) {
            continue;
        }

        // NOTE: Tile is a special case
        if (child->isTiled()) {
            auto& am = s.getAttrManager();
            auto params = std::any_cast<TileParams>(am.parseAttr(*child->attr, s).value());

            auto [firstMeta, secondMeta] = splitTileAttr(*child, r);
            //  if (metadata.type.size() > 0)
            {
                auto loopType = child->type.front();
                if (loopType == LoopType::Outer) {
                    outer.push_back(firstMeta);
                } else if (loopType == LoopType::Inner) {
                    inner.push_back(firstMeta);
                }
            }

            if (child->type.size() > 1) {
                auto loopType = child->type[1];
                if (loopType == LoopType::Outer) {
                    outer.push_back(secondMeta);
                } else if (loopType == LoopType::Inner) {
                    inner.push_back(secondMeta);
                }
            }

            continue;
        }

        auto metadata = LoopMetaData(*child, r);
        if (child->is(LoopType::Outer)) {
            outer.emplace_back(std::move(metadata));
            continue;
        }

        if (child->is(LoopType::Inner)) {
            inner.emplace_back(std::move(metadata));
            continue;
        }
    }

    // Declare loop data
    out << "occa::dim outer, inner;\n";
    out << "outer.dims = " << outer.size() << ";\n";
    out << "inner.dims = " << inner.size() << ";\n";

    // Outer loops
    {
        auto n = outer.size();
        for (auto& loop : outer) {
            --n;
            out << getLoopInfoStr(loop, n, true);
        }
    }

    // Inner loops
    {
        auto n = inner.size();
        for (auto& loop : inner) {
            --n;
            out << getLoopInfoStr(loop, n, false);
        }
    }

    out << "occa::kernel kernel(deviceKernels[" << loopNo << "]);\n";
    out << "kernel.setRunDims(outer, inner);\n";

    // Kernel call
    out << "kernel";
    out << "(";
    {
        for (auto it = decl.param_begin(), end = decl.param_end(); it != end; ++it) {
            auto* param = *it;
            if (!param) {
                continue;
            }
            out << (it != decl.param_begin() ? ", " : "") << param->getNameAsString();
        }
    }
    out << ");\n";

    out << "};\n";

    return out.str();
}

HandleResult handleLauncherTranslationUnit(const TranslationUnitDecl& d, SessionStage& s) {
    SPDLOG_DEBUG("Handle translation unit");

    auto& backendDeps = s.tryEmplaceUserCtx<HeaderDepsInfo>().backendHeaders;
    backendDeps.clear();
    backendDeps.emplace_back("#include " + std::string(includeOCCA) + "\n\n");

    return {};
}

HandleResult handleLauncherKernelAttribute(const Attr& a,
                                           const FunctionDecl& func,
                                           SessionStage& s) {
    SPDLOG_DEBUG("Handle attribute: {}", a.getNormalizedFullName());

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto& rewriter = s.getRewriter();

    if (!sema.getParsingKernelInfo()) {
        return tl::make_unexpected(
            Error{OkltPipelineErrorCode::INTERNAL_ERROR_KERNEL_INFO_NULL, "handleKernelAttribute"});
    }

    auto kernelInfo = *sema.getParsingKernelInfo();
    auto& kernels = sema.getProgramMetaData().kernels;

    auto& meta = kernels.emplace_back();
    meta.name = func.getNameAsString();

    // Add 'extern "C"'
    rewriter.ReplaceText(getAttrFullSourceRange(a), externC);

    auto paramStr = getFunctionParamStr(func, meta);
    auto typeLoc = func.getFunctionTypeLoc();
    auto paramsRange = SourceRange(typeLoc.getLParenLoc().getLocWithOffset(1),
                                   typeLoc.getRParenLoc().getLocWithOffset(-1));
    rewriter.ReplaceText(paramsRange, paramStr);

    size_t n = 0;
    for (auto* loop : kernelInfo.topLevelOuterLoops) {
        if (!loop) {
            continue;
        }
        removeAttribute(*loop->attr, s);

        auto body = getRootLoopBody(func, *loop, n, s);
        // NOTE: rewriter order matter! First get body, then remove, otherwise UB !!!
        rewriter.RemoveText(SourceRange{loop->stmt.getForLoc(), loop->stmt.getRParenLoc()});
        rewriter.ReplaceText(loop->stmt.getBody()->getSourceRange(), body);
        ++n;
    }

    return {};
}

__attribute__((constructor)) void registerLauncherHandler() {
#define REG_ATTR_HANDLE(NAME, BODY)                                                   \
    {                                                                                 \
        auto ok = oklt::AttributeManager::instance().registerBackendHandler(          \
            {TargetBackend::_LAUNCHER, NAME}, BODY);                                  \
        if (!ok) {                                                                    \
            SPDLOG_ERROR("Failed to register {} attribute handler (Launcher)", NAME); \
        }                                                                             \
    }

#define REG_IMPLICIT_HANDLE(KIND, BODY)                                                       \
    {                                                                                         \
        auto ok = oklt::AttributeManager::instance().registerImplicitHandler(                 \
            {TargetBackend::_LAUNCHER, KIND}, BODY);                                          \
        if (!ok) {                                                                            \
            SPDLOG_ERROR("Failed to register {} attribute handler (Launcher)", (size_t)KIND); \
        }                                                                                     \
    }

    REG_IMPLICIT_HANDLE(clang::Decl::Kind::TranslationUnit,
                        makeSpecificImplicitHandle(handleLauncherTranslationUnit));

    REG_ATTR_HANDLE(KERNEL_ATTR_NAME, makeSpecificAttrHandle(handleLauncherKernelAttribute));
    REG_ATTR_HANDLE(OUTER_ATTR_NAME, AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});
    REG_ATTR_HANDLE(INNER_ATTR_NAME, AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});
    REG_ATTR_HANDLE(TILE_ATTR_NAME, AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});

    REG_ATTR_HANDLE(ATOMIC_ATTR_NAME, AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});
    REG_ATTR_HANDLE(BARRIER_ATTR_NAME, AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});
    REG_ATTR_HANDLE(EXCLUSIVE_ATTR_NAME, AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});
    REG_ATTR_HANDLE(EXCLUSIVE_ATTR_NAME, AttrDeclHandler{serial_subset::handleEmptyDeclAttribute});
    REG_ATTR_HANDLE(SHARED_ATTR_NAME, AttrDeclHandler{serial_subset::handleEmptyDeclAttribute});
    REG_ATTR_HANDLE(SHARED_ATTR_NAME, AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});

    REG_ATTR_HANDLE(RESTRICT_ATTR_NAME,
                    makeSpecificAttrHandle(serial_subset::handleRestrictAttribute));

#undef REG_ATTR_HANDLE
}
}  // namespace
