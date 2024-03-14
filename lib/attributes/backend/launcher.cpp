#include "attributes/attribute_names.h"
#include "attributes/frontend/params/loop.h"
#include "attributes/frontend/params/tile.h"
#include "attributes/utils/serial_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/type_converter.h"

// #define OKL_LAUNCHER_RECURSIVE

namespace {
using namespace oklt;
using namespace clang;

const std::string includeOCCA = "<occa/core/kernel.hpp>";
const std::string externC = "extern \"C\"";

std::string getTiledVariableName(const OklLoopInfo& forLoop) {
    auto& meta = forLoop.metadata;
    return "_occa_tiled_" + meta.var.name;
}

// TODO: Replace with ArgumentInfo::toString()
std::string getFunctionDeclParamsStr(const FunctionDecl& decl, KernelInfo& kernelInfo) {
    std::stringstream out;
    // out << "(";

    kernelInfo.args.clear();

    kernelInfo.args.emplace_back(ArgumentInfo{.is_const = false,
                                              .dtype = DataType{.type = DatatypeCategory::CUSTOM},
                                              .name = "deviceKernels",
                                              .is_ptr = true});
    out << util::fmt("{} {} {}", "occa::modeKernel_t", "**", "deviceKernels").value();

    for (auto p : decl.parameters()) {
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
                             .dtype = DataType{.type = DatatypeCategory::CUSTOM},
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

std::pair<LoopMetaData, LoopMetaData> splitTileAttr(OklLoopInfo& loopInfo, std::string& tileSize) {
    auto sz = util::parseStrTo<size_t>(tileSize);

    // Prepare first loop
    LoopMetaData firstMeta = loopInfo.metadata;
    firstMeta.var.name = getTiledVariableName(loopInfo);
    if (sz.value_or(1024) > 1) {
        if (firstMeta.inc.val.empty()) {
            firstMeta.inc.val = tileSize;
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
            firstMeta.inc.val = "(" + tileSize + " * " + firstMeta.inc.val + ")";
        }
    }

    // Prepare second loop
    LoopMetaData secondMeta = loopInfo.metadata;
    secondMeta.range.start = firstMeta.var.name;
    switch (secondMeta.condition.op) {
        case BinOp::Le:
            secondMeta.condition.op = BinOp::Lt;
            break;
        case BinOp::Ge:
            secondMeta.condition.op = BinOp::Gt;
            break;
    }
    if (sz.value_or(1024) > 1) {
        secondMeta.range.end = "(" + firstMeta.var.name + " + " + tileSize + ")";
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
    out << " {\n";

    // List all loops
    std::list<OklLoopInfo*> loops = {};
    collectLoops(loopInfo, loops);

    // Prepare metadata for outer and inner loops
    std::list<LoopMetaData> outer = {};
    std::list<LoopMetaData> inner = {};
    for (auto child : loops) {
        auto& metadata = child->metadata;
        if (metadata.type.empty()) {
            continue;
        }

        // NOTE: Tile is a special case
        if (child->isTiled()) {
            auto& am = s.getAttrManager();
            auto params = std::any_cast<TileParams>(am.parseAttr(child->attr, s).value());

            auto [firstMeta, secondMeta] = splitTileAttr(*child, params.tileSize);
            //  if (metadata.type.size() > 0)
            {
                auto loopType = metadata.type.front();
                if (loopType == LoopType::Outer) {
                    outer.push_back(firstMeta);
                } else if (loopType == LoopType::Inner) {
                    inner.push_back(firstMeta);
                }
            }

            if (metadata.type.size() > 1) {
                auto loopType = metadata.type[1];
                if (loopType == LoopType::Outer) {
                    outer.push_back(secondMeta);
                } else if (loopType == LoopType::Inner) {
                    inner.push_back(secondMeta);
                }
            }

            continue;
        }

        if (child->is(LoopType::Outer)) {
            outer.push_back(metadata);
            continue;
        }

        if (child->is(LoopType::Inner)) {
            inner.push_back(metadata);
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
        out << "deviceKernels";
        for (auto param : decl.parameters()) {
            if (!param) {
                continue;
            }
            out << ", " << param->getNameAsString();
        }
    }
    out << ");\n";

    out << "};\n";

    return out.str();
}

HandleResult handleLauncherTranslationUnit(const TranslationUnitDecl& d, SessionStage& s) {
    auto& sm = s.getCompiler().getSourceManager();
    auto mainFileId = sm.getMainFileID();
    auto loc = sm.getLocForStartOfFile(mainFileId);

#ifdef TRANSPILER_DEBUG_LOG
    auto offset = sm.getFileOffset(d.getLocation());
    llvm::outs() << "[DEBUG] Found translation unit, offset: " << offset << "\n";
#endif

    //    s.getRewriter().InsertTextBefore(loc, "#include " + includeOCCA + "\n\n");
    auto& backendDeps = s.tryEmplaceUserCtx<HeaderDepsInfo>().backendDeps;
    backendDeps.clear();
    backendDeps.emplace_back("#include " + std::string(includeOCCA) + "\n\n");

    return {};
}
HandleResult handleLauncherKernelAttribute(const Attr& a,
                                           const FunctionDecl& decl,
                                           SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto& rewriter = s.getRewriter();

    // Add 'extern "C"'
    rewriter.ReplaceText(getAttrFullSourceRange(a), externC);

    auto kernelInfo = sema.getParsingKernelInfo();
    if (!kernelInfo || !kernelInfo->kernInfo) {
        return {};
    }

    kernelInfo->kernInfo->name = decl.getNameAsString();

    auto paramsStr = getFunctionDeclParamsStr(decl, *kernelInfo->kernInfo);
    rewriter.ReplaceText(decl.getParametersSourceRange(), paramsStr);

    size_t n = 0;
    for (auto& loop : kernelInfo->children) {
        removeAttribute(loop.attr, s);
        rewriter.RemoveText(SourceRange{loop.stmt.getForLoc(), loop.stmt.getRParenLoc()});

        auto body = getRootLoopBody(decl, loop, n, s);
        rewriter.ReplaceText(loop.stmt.getBody()->getSourceRange(), body);
        ++n;
    }

    return {};
}

__attribute__((constructor)) void registerLauncherHandler() {
#define REG_ATTR_HANDLE(NAME, BODY)                                                             \
    {                                                                                           \
        auto ok = oklt::AttributeManager::instance().registerBackendHandler(                    \
            {TargetBackend::_LAUNCHER, NAME}, BODY);                                            \
        if (!ok) {                                                                              \
            llvm::errs() << "failed to register " << NAME << " attribute handler (Launcher)\n"; \
        }                                                                                       \
    }

    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::_LAUNCHER, clang::Decl::Kind::TranslationUnit},
        makeSpecificImplicitHandle(handleLauncherTranslationUnit));

    if (!ok) {
        llvm::errs() << "Failed to register implicit handler for translation unit (Launcher)\n";
    }

    REG_ATTR_HANDLE(KERNEL_ATTR_NAME, makeSpecificAttrHandle(handleLauncherKernelAttribute));
    REG_ATTR_HANDLE(OUTER_ATTR_NAME, AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});
    REG_ATTR_HANDLE(INNER_ATTR_NAME, AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});
    REG_ATTR_HANDLE(TILE_ATTR_NAME, AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});

    REG_ATTR_HANDLE(ATOMIC_ATTR_NAME, AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});
    REG_ATTR_HANDLE(BARRIER_ATTR_NAME, AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});
    REG_ATTR_HANDLE(EXCLUSIVE_ATTR_NAME, AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});
    REG_ATTR_HANDLE(EXCLUSIVE_ATTR_NAME, AttrDeclHandler{serial_subset::handleEmptyDeclAttribute});
    REG_ATTR_HANDLE(SHARED_ATTR_NAME, AttrDeclHandler{serial_subset::handleEmptyDeclAttribute});

    REG_ATTR_HANDLE(RESTRICT_ATTR_NAME,
                    makeSpecificAttrHandle(serial_subset::handleRestrictAttribute));

#undef REG_ATTR_HANDLE
}
}  // namespace
