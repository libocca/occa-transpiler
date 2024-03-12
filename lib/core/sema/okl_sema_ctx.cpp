#include "attributes/attribute_names.h"
#include "attributes/frontend/params/tile.h"

#include "core/sema/okl_sema_ctx.h"
#include "core/utils/ast_node_parsers.h"
#include "core/utils/type_converter.h"

#include <oklt/core/error.h>
#include <oklt/util/string_utils.h>

#include <clang/AST/Attr.h>
#include <deque>

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
                                                 OklSemaCtx::ParsedKernelInfo& kernelInfo) {
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

bool OklSemaCtx::startParsingOklKernel(const FunctionDecl& fd) {
    if (_parsingKernInfo) {
        return false;
    }

    // create slot for kernel info in list
    auto* kiPtr = &_programMetaData.addKernelInfo(fd.getNameAsString(), fd.param_size());
    auto kiParams = std::vector<std::string>(fd.param_size());
    _parsingKernInfo = &_parsedKernelList.emplace_back(fd, std::move(kiParams), kiPtr);

    // Set kernel info parameters
    for (auto param : fd.parameters()) {
        if (param) {
            setKernelArgInfo(*param);
            setTranspiledArgStr(*param);
        }
    }

    return true;
}

OklSemaCtx::ParsedKernelInfo* OklSemaCtx::getParsingKernelInfo() {
    return _parsingKernInfo;
}

void OklSemaCtx::setParsedKernelInfo(ParsedKernelInfo* ki) {
    _parsingKernInfo = ki;
}

void OklSemaCtx::stopParsingKernelInfo() {
    _parsingKernInfo = nullptr;
}

bool OklSemaCtx::isParsingOklKernel() const {
    return _parsingKernInfo;
}

bool OklSemaCtx::isCurrentParsingOklKernel(const clang::FunctionDecl& fd) const {
    if (!_parsingKernInfo) {
        return false;
    }

    return std::addressof(_parsingKernInfo->decl.get()) == std::addressof(fd);
}

bool OklSemaCtx::isDeclInLexicalTraversal(const Decl& decl) const {
    if (!_parsingKernInfo) {
        return false;
    }

    return cast<FunctionDecl>(decl.getParentFunctionOrMethod()) ==
           std::addressof(_parsingKernInfo->decl.get());
}

OklLoopInfo* OklSemaCtx::getLoopInfo(const clang::ForStmt& forStmt) const {
    if (!_parsingKernInfo) {
        return nullptr;
    }

    auto it = _parsingKernInfo->loopMap.find(&forStmt);
    if (it == _parsingKernInfo->loopMap.end() || !it->second) {
        return nullptr;
    }

    return it->second;
}

[[nodiscard]] OklLoopInfo* OklSemaCtx::getLoopInfo() {
    if (!_parsingKernInfo) {
        return nullptr;
    }

    return _parsingKernInfo->currentLoop;
}

void OklSemaCtx::setLoopInfo(OklLoopInfo* loopInfo) {
    if (!_parsingKernInfo) {
        return;
    }

    auto it = std::find_if(_parsingKernInfo->loopMap.begin(),
                           _parsingKernInfo->loopMap.end(),
                           [loopInfo](const auto& v) { return v.second == loopInfo; });
    if (it != _parsingKernInfo->loopMap.end()) {
        _parsingKernInfo->currentLoop = loopInfo;
    }
}

tl::expected<void, Error> OklSemaCtx::startParsingAttributedForLoop(const clang::Attr& attr,
                                                                    const clang::ForStmt& stmt,
                                                                    const std::any* params) {
    assert(_parsingKernInfo);
    auto loopType = getLoopType(params);

    if (!_parsingKernInfo->currentLoop && !isLegalTopLoopLevel(loopType)) {
        return tl::make_unexpected(Error{
            .ec = std::error_code(), .desc = "[@kernel] requires at least one [@outer] for-loop"});
    }

    auto currentLoop = _parsingKernInfo->currentLoop;
    auto& children = currentLoop ? currentLoop->children : _parsingKernInfo->children;
    auto parentType = currentLoop ? currentLoop->metadata.type : LoopMetaType::Regular;

    if (!isLegalLoopLevel(loopType, parentType)) {
        return tl::make_unexpected(
            Error{.ec = std::error_code(),
                  .desc = "Cannot have [@inner] loop outside of an [@outer] loop"});
    }

    return makeOklLoopInfo(stmt, attr, loopType, *_parsingKernInfo)
        .and_then([&children, this](auto&& loopInfo) -> tl::expected<void, Error> {
            children.emplace_back(loopInfo);

            auto& child = children.back();
            child.parent = _parsingKernInfo->currentLoop;
            _parsingKernInfo->currentLoop = &child;
            _parsingKernInfo->loopMap.emplace(&child.stmt, &child);
            return {};
        });
}

tl::expected<void, Error> OklSemaCtx::stopParsingAttributedForLoop(const clang::Attr& attr,
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

    _parsingKernInfo->currentLoop = loopInfo->parent;

    return {};
}

void OklSemaCtx::setKernelArgInfo(const ParmVarDecl& parm) {
    assert(_parsingKernInfo);
    auto result = toOklArgInfo(parm);
    if (!result) {
        llvm::errs() << "failed to convert parm var decl to okl data type\n";
        return;
    }

    auto* ki = _parsingKernInfo->kernInfo;
    ki->args[parm.getFunctionScopeIndex()] = std::move(result.value());
}

void OklSemaCtx::setTranspiledArgStr(const ParmVarDecl& parm, std::string_view transpiledArgStr) {
    assert(_parsingKernInfo);
    if (!transpiledArgStr.empty()) {
        auto& pki = *_parsingKernInfo;
        pki.argStrs[parm.getFunctionScopeIndex()] = std::string(transpiledArgStr);
    }

    auto& pki = *_parsingKernInfo;
    pki.argStrs[parm.getFunctionScopeIndex()] =
        parm.getType().getAsString() + " " + parm.getNameAsString();
}

void OklSemaCtx::setKernelTranspiledAttrStr(std::string attrStr) {
    assert(_parsingKernInfo);
    _parsingKernInfo->transpiledFuncAttrStr = std::move(attrStr);
}

ProgramMetaData& OklSemaCtx::getProgramMetaData() {
    return _programMetaData;
}
const ProgramMetaData& OklSemaCtx::getProgramMetaData() const {
    return _programMetaData;
}
}  // namespace oklt
