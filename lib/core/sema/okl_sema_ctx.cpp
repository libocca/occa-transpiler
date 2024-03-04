#include "attributes/frontend/params/tile.h"

#include "core/sema/okl_sema_ctx.h"
#include "core/utils/ast_node_parsers.h"
#include "core/utils/type_converter.h"

#include <oklt/core/error.h>
#include <oklt/util/string_utils.h>

#include <clang/AST/Attr.h>

namespace {
using namespace clang;
using namespace oklt;

LoopMetaType getLoopType(const std::any* param) {
    if (!param) {
        return LoopMetaType::Regular;
    }

    if (param->type() == typeid(TileParams)) {
        auto tile = std::any_cast<TileParams>(*param);
        if (tile.firstLoop.type == AttributedLoopType::Inner) {
            return LoopMetaType::Inner;
        }
        if (tile.firstLoop.type == AttributedLoopType::Outer) {
            if (tile.secondLoop.type == AttributedLoopType::Inner) {
                return LoopMetaType::OuterInner;
            }
            return LoopMetaType::Outer;
        }
    } else if (param->type() == typeid(AttributedLoop)) {
        auto loop = std::any_cast<AttributedLoop>(*param);
        if (loop.type == AttributedLoopType::Outer) {
            return LoopMetaType::Outer;
        } else if (loop.type == AttributedLoopType::Inner) {
            return LoopMetaType::Inner;
        }
    }

    return LoopMetaType::Regular;
}

tl::expected<OklLoopInfo, Error> makeOklLoopInfo(const clang::ForStmt& stmt,
                                                 const clang::Attr& attr,
                                                 LoopMetaType loopType,
                                                 OklSemaCtx::ParsingKernelInfo& kernelInfo) {
    assert(kernelInfo.kernInfo);

    auto parsedLoopMeta = parseForStmt(stmt, kernelInfo.decl.get().getASTContext());
    if (!parsedLoopMeta) {
        return tl::make_unexpected(std::move(parsedLoopMeta.error()));
    }

    auto& metaList = kernelInfo.currentLoop ? kernelInfo.currentLoop->metadata.childrens
                                            : kernelInfo.kernInfo->childrens;
    metaList.emplace_back(std::move(parsedLoopMeta.value()));

    auto ret = OklLoopInfo{.attr = attr, .stmt = stmt, .metadata = metaList.back()};
    ret.metadata.type = loopType;
    return ret;
}

bool isLegalLoopLevel(LoopMetaType childType, LoopMetaType parentType = LoopMetaType::Regular) {
    if (parentType == LoopMetaType::OuterInner) {
        parentType = LoopMetaType::Inner;
    }
    if (childType == LoopMetaType::OuterInner) {
        childType = LoopMetaType::Outer;
    } else if (childType == LoopMetaType::Regular) {
        childType = parentType;
    }
    return parentType == LoopMetaType::Regular || parentType == childType ||
           (parentType == LoopMetaType::Outer && childType == LoopMetaType::Inner);
}

bool isLegalTopLoopLevel(LoopMetaType loopType) {
    return loopType == LoopMetaType::OuterInner || loopType == LoopMetaType::Outer;
}

}  // namespace

namespace oklt {
OklSemaCtx::ParsingKernelInfo* OklSemaCtx::startParsingOklKernel(const FunctionDecl& fd) {
    if (_parsingKernInfo.has_value()) {
        return nullptr;
    }

    auto* kiPtr = &_programMetaData.addKernelInfo(fd.getNameAsString(), fd.param_size(), 1);
    auto kiParams = std::vector<std::string>(fd.param_size());
    _parsingKernInfo.emplace(fd, std::move(kiParams), kiPtr);

    return &_parsingKernInfo.value();
}

OklSemaCtx::ParsingKernelInfo* OklSemaCtx::getParsingKernelInfo() {
    return _parsingKernInfo ? &_parsingKernInfo.value() : nullptr;
}

void OklSemaCtx::stopParsingKernelInfo() {
    _parsingKernInfo.reset();
}

bool OklSemaCtx::isParsingOklKernel() const {
    return _parsingKernInfo.has_value();
}

bool OklSemaCtx::isCurrentParsingOklKernel(const clang::FunctionDecl& fd) const {
    if (!_parsingKernInfo) {
        return false;
    }
    auto& kernelInfo = _parsingKernInfo.value();

    return std::addressof(kernelInfo.decl.get()) == std::addressof(fd);
}

bool OklSemaCtx::isDeclInLexicalTraversal(const Decl& decl) const {
    if (!_parsingKernInfo.has_value()) {
        return false;
    }
    auto& kernelInfo = _parsingKernInfo.value();

    return cast<FunctionDecl>(decl.getParentFunctionOrMethod()) ==
           std::addressof(kernelInfo.decl.get());
}

std::optional<OklLoopInfo> OklSemaCtx::getLoopInfo(const clang::ForStmt& forStmt) const {
    if (!_parsingKernInfo) {
        return std::nullopt;
    }
    auto& kernelInfo = _parsingKernInfo.value();

    auto it = kernelInfo.loopMap.find(&forStmt);
    if (it == kernelInfo.loopMap.end() || !it->second) {
        return std::nullopt;
    }

    return *it->second;
}

[[nodiscard]] std::optional<OklLoopInfo> OklSemaCtx::getLoopInfo() {
    if (!_parsingKernInfo) {
        return std::nullopt;
    }

    auto& kernelInfo = _parsingKernInfo.value();
    if (kernelInfo.currentLoop) {
        return *kernelInfo.currentLoop;
    }

    return std::nullopt;
}

tl::expected<void, Error> OklSemaCtx::validateOklForLoopOnPreTraverse(const clang::Attr& attr,
                                                                      const clang::ForStmt& stmt,
                                                                      const std::any* params) {
    assert(_parsingKernInfo);

    auto loopType = getLoopType(params);
    //    if (loopType == LoopMetaType::Regular) {
    //        return {};
    //    }

    auto& kernelInfo = _parsingKernInfo.value();

    if (!kernelInfo.currentLoop && !isLegalTopLoopLevel(loopType)) {
        return tl::make_unexpected(Error{
            .ec = std::error_code(), .desc = "[@kernel] requires at least one [@outer] for-loop"});
    }

    auto& children =
        kernelInfo.currentLoop ? kernelInfo.currentLoop->children : kernelInfo.children;
    auto parentType =
        kernelInfo.currentLoop ? kernelInfo.currentLoop->metadata.type : LoopMetaType::Regular;

    if (!isLegalLoopLevel(loopType, parentType)) {
        return tl::make_unexpected(
            Error{.ec = std::error_code(),
                  .desc = "Cannot have [@inner] loop outside of an [@outer] loop"});
    }

    return makeOklLoopInfo(stmt, attr, loopType, kernelInfo)
        .and_then([&children, &kernelInfo](auto&& loopInfo) -> tl::expected<void, Error> {
            children.emplace_back(loopInfo);

            auto& child = children.back();
            child.parent = kernelInfo.currentLoop;
            kernelInfo.currentLoop = &child;
            kernelInfo.loopMap.emplace(&child.stmt, &child);
            return {};
        });
}

tl::expected<void, Error> OklSemaCtx::validateOklForLoopOnPostTraverse(const clang::Attr& attr,
                                                                       const clang::ForStmt& stmt,
                                                                       const std::any* params) {
    assert(_parsingKernInfo);

    auto loopType = getLoopType(params);
    if (loopType == LoopMetaType::Regular) {
        return {};
    }

    auto loopInfo = getLoopInfo(stmt);
    if (!loopInfo) {
        return {};
    }

    auto& kernelInfo = _parsingKernInfo.value();
    kernelInfo.currentLoop = loopInfo->parent;

    return {};
}

void OklSemaCtx::setKernelArgInfo(const ParmVarDecl& parm) {
    assert(_parsingKernInfo.has_value());
    auto result = toOklArgInfo(parm);
    if (!result) {
        llvm::errs() << "failed to convert parm var decl to okl data type\n";
        return;
    }

    auto* ki = _parsingKernInfo.value().kernInfo;
    ki->args[parm.getFunctionScopeIndex()] = std::move(result.value());
}

void OklSemaCtx::setTranspiledArgStr(const ParmVarDecl& parm, std::string_view transpiledArgStr) {
    assert(_parsingKernInfo.has_value());
    if (!transpiledArgStr.empty()) {
        auto& pki = _parsingKernInfo.value();
        pki.argStrs[parm.getFunctionScopeIndex()] = std::string(transpiledArgStr);
    }

    auto& pki = _parsingKernInfo.value();
    pki.argStrs[parm.getFunctionScopeIndex()] =
        parm.getType().getAsString() + " " + parm.getNameAsString();
}

void OklSemaCtx::setKernelTranspiledAttrStr(std::string attrStr) {
    assert(_parsingKernInfo.has_value());
    _parsingKernInfo.value().transpiledFuncAttrStr = std::move(attrStr);
}

ProgramMetaData& OklSemaCtx::getProgramMetaData() {
    return _programMetaData;
}
const ProgramMetaData& OklSemaCtx::getProgramMetaData() const {
    return _programMetaData;
}
}  // namespace oklt
